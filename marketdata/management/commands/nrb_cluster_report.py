import csv
import os
from datetime import datetime
from django.core.management.base import BaseCommand
from core.pattern_recognition import get_pattern_triggers

class Command(BaseCommand):
    help = 'Generates an RSC Report that EXACTLY matches the Frontend Graph Logic'

    def handle(self, *args, **options):
        # 1. Define the list of stocks
        target_stocks = [
            "CUB", "NYKAA", "TITAN", "FORCEMOT", "CUPID", "DIXON", 
            "LAURUSLABS", "AMBER", "CUMMINSIND", "BAJFINANCE", "INDIGO"
        ]

        # =========================================================================
        # ⚙️ CONFIGURATION (Matches your App.tsx defaults)
        # =========================================================================
        # "52 is the size of the rolling window" - We set this to 52.
        # The function will scan the entire database history using this window.
        PATTERN_WINDOW_WEEKS = 52       
        COOLDOWN_WEEKS = 5    
        SERIES_NAME = "rsc_sensex_ratio"
        # =========================================================================

        print(f"Generating RSC Sensex Report (Window: {PATTERN_WINDOW_WEEKS}w, Cooldown: {COOLDOWN_WEEKS}w)...")
        print("-" * 130)
        
        headers = [
            "Stock", "RSC Level", "NRB Count", "Start Date", "End Date", 
            "Duration (wks)", "3M %", "6M %", "12M %", 
            "90-95%", "95-98%", "98-100%" 
        ]

        row_format = "{:<12} {:<14} {:<10} {:<12} {:<12} {:<15} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}"
        print(row_format.format(*headers))
        print("-" * 130)

        report_data = []

        for symbol in target_stocks:
            try:
                # 2. Fetch Data (Full History scan with 52-week window)
                triggers = get_pattern_triggers(
                    scrip=symbol,
                    pattern="Narrow Range Break",
                    nrb_lookback=7,
                    success_rate=0,
                    weeks=PATTERN_WINDOW_WEEKS,     # 52 Weeks Rolling Window
                    series=SERIES_NAME,      
                    cooldown_weeks=COOLDOWN_WEEKS   
                )

                if not triggers:
                    continue

                # 3. Group Triggers (Raw Data)
                groups = {}
                for t in triggers:
                    gid = t.get('nrb_group_id')
                    if not gid: continue
                    if gid not in groups:
                        groups[gid] = t

                # ---------------------------------------------------------
                # 🟢 LOGIC PORT: EXACT MATCH OF App.tsx
                # ---------------------------------------------------------
                
                # Step A: Identify "Historical Candidates" (Duration > 24)
                # These are the BLUE lines on your chart.
                historical_candidates = []
                
                for gid, t in groups.items():
                    # TRUST THE BACKEND DURATION
                    # This value comes directly from the DB/Backend logic used by the graph.
                    dur_val = t.get('group_duration_weeks')
                    
                    if dur_val is not None:
                        duration_val = float(dur_val)
                    else:
                        # Fallback only if missing
                        start_ts = t.get('group_start_time')
                        end_ts = t.get('group_end_time')
                        if start_ts and end_ts:
                            days = (datetime.fromtimestamp(end_ts) - datetime.fromtimestamp(start_ts)).days
                            duration_val = round(days / 7.0, 1)
                        else:
                            duration_val = 0

                    if duration_val > 24:
                        t['group_duration_weeks'] = duration_val 
                        historical_candidates.append(t)

                # Step B: Sort by Level Descending (Highest Ratio First)
                historical_candidates.sort(key=lambda x: float(x.get('group_level', 0)), reverse=True)

                # Step C: Apply Visibility Filter (Hide lower overlapping lines)
                # This ensures we don't show lines that are hidden behind "King of the Hill" logic
                visible_historical = []
                TIME_BUFFER = 365 * 24 * 60 * 60 # 365 Days overlap buffer

                for candidate in historical_candidates:
                    start_A = candidate.get('group_start_time')
                    end_A = candidate.get('group_end_time')
                    
                    if not start_A or not end_A: continue

                    is_hidden = False

                    # Check against lines we have ALREADY accepted (which are higher level)
                    for existing in visible_historical:
                        start_B = existing.get('group_start_time')
                        end_B = existing.get('group_end_time')

                        # Check Time Overlap
                        is_overlapping = (start_A < (end_B + TIME_BUFFER)) and (start_B < (end_A + TIME_BUFFER))

                        if is_overlapping:
                            is_hidden = True
                            break 
                    
                    if not is_hidden:
                        visible_historical.append(candidate)

                # 4. Generate Rows
                for t in visible_historical:
                    level_val = float(t.get('group_level', 0))
                    duration_val = float(t.get('group_duration_weeks', 0))
                    start_ts = t.get('group_start_time')
                    end_ts = t.get('group_end_time')
                    
                    row = [
                        symbol, 
                        f"{level_val:.4f}",
                        t.get('group_nrb_count', 0),
                        datetime.fromtimestamp(start_ts).strftime('%Y-%m-%d'),
                        datetime.fromtimestamp(end_ts).strftime('%Y-%m-%d'),
                        f"{duration_val:.1f}", 
                        f"+{t.get('zone_success_rate_3m')}" if (t.get('zone_success_rate_3m') or 0) > 0 else str(t.get('zone_success_rate_3m') or "-"),
                        f"+{t.get('zone_success_rate_6m')}" if (t.get('zone_success_rate_6m') or 0) > 0 else str(t.get('zone_success_rate_6m') or "-"),
                        f"+{t.get('zone_success_rate_12m')}" if (t.get('zone_success_rate_12m') or 0) > 0 else str(t.get('zone_success_rate_12m') or "-"),
                    ]
                    
                    c98, c95, c90 = 0, 0, 0
                    if 'near_touches' in t and t['near_touches']:
                        for touch in t['near_touches']:
                            diff = touch.get('min_diff_pct', 100)
                            if diff < 2.0: c98 += 1
                            elif diff < 5.0: c95 += 1
                            elif diff < 10.0: c90 += 1
                    
                    row.extend([c90, c95, c98])
                    report_data.append(row)
                    print(row_format.format(*[str(x) for x in row]))

            except Exception as e:
                print(f"Error processing {symbol}: {e}")

        # 5. Export
        csv_filename = "nrb_rsc_final_report.csv"
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(report_data)

        print("-" * 130)
        print(f"Done! Report saved to {os.path.abspath(csv_filename)}")