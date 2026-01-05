import csv
import os
from datetime import datetime
from django.core.management.base import BaseCommand
from core.pattern_recognition import get_pattern_triggers

class Command(BaseCommand):
    help = 'Generates the NRB Report with Local Duration (Start to Break)'

    def handle(self, *args, **options):
        # 1. Define the list of stocks
        target_stocks = [
            "CUB",          # City Union Bank
            "NYKAA",        # Nykaa
            "TITAN",        # Titan
            "FORCEMOT",     # Force Motors
            "CUPID",        # Cupid
            "DIXON",        # Dixon
            "LAURUSLABS",   # Laurus
            "AMBER",        # Amber
            "CUMMINSIND",   # Cummins
            "BAJFINANCE",   # Bajaj Finance
            "INDIGO"        # Interglobe (Indigo)
        ]

        print(f"Generating NRB Local Duration Report (52 Weeks / 5 Cooldown) for {len(target_stocks)} stocks...")
        print("-" * 130)
        
        # 2. Define Headers
        headers = [
            "Stock", 
            "NRB Start",     # Start of the narrow range (Lookback start)
            "Break Date",    # The 'Arrow' date
            "Duration (wks)",# Local duration (Break - Start)
            "Range %",       # Tightness of the range
            "3M %", "6M %", "12M %", # Future performance
            "90-95%", "95-98%", "98-100%" # Attempt Counts
        ]

        row_format = "{:<12} {:<12} {:<12} {:<15} {:<10} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}"
        print(row_format.format(*headers))
        print("-" * 130)

        report_data = []

        for symbol in target_stocks:
            try:
                # 3. Fetch Data
                triggers = get_pattern_triggers(
                    scrip=symbol,
                    pattern="Narrow Range Break",
                    nrb_lookback=7,   
                    success_rate=0,
                    weeks=52,         # 🟢 Scan Window
                    series="close",
                    cooldown_weeks=5  # 🟢 Cooldown
                )

                if not triggers:
                    continue

                for t in triggers:
                    # 4. Extract Data
                    start_ts = t.get('range_start_time')
                    break_ts = t.get('time') # The 'Arrow' Time
                    
                    if not start_ts or not break_ts:
                        continue

                    start_date = datetime.fromtimestamp(start_ts).strftime('%Y-%m-%d')
                    break_date = datetime.fromtimestamp(break_ts).strftime('%Y-%m-%d')

                    # 🆕 UPDATED: Force Local Duration Calculation
                    # Always calculate: (Break Date - Start Date) / 7
                    # This ignores the 'Master Zone' duration (e.g. 106) from the backend.
                    days_diff = (datetime.fromtimestamp(break_ts) - datetime.fromtimestamp(start_ts)).days
                    duration_wks = round(days_diff / 7.0, 1)

                    # Range %
                    r_high = float(t.get('range_high', 0))
                    r_low = float(t.get('range_low', 0))
                    range_pct = round(((r_high - r_low) / r_low) * 100, 2) if r_low > 0 else 0.0

                    # Success Rates (Keep from Zone if available)
                    s3m = t.get('zone_success_rate_3m')
                    s6m = t.get('zone_success_rate_6m')
                    s12m = t.get('zone_success_rate_12m')

                    str_3m = f"+{s3m}" if s3m and s3m > 0 else str(s3m or "-")
                    str_6m = f"+{s6m}" if s6m and s6m > 0 else str(s6m or "-")
                    str_12m = f"+{s12m}" if s12m and s12m > 0 else str(s12m or "-")

                    # Attempts Counts
                    count_90_95 = 0
                    count_95_98 = 0
                    count_98_100 = 0

                    if 'near_touches' in t and t['near_touches']:
                        for touch in t['near_touches']:
                            diff = touch.get('min_diff_pct', 100)
                            if diff < 2.0:
                                count_98_100 += 1
                            elif diff < 5.0:
                                count_95_98 += 1
                            elif diff < 10.0:
                                count_90_95 += 1

                    # 5. Create Row
                    row = [
                        symbol, 
                        start_date, 
                        break_date, 
                        duration_wks, # Now strictly Local Duration
                        f"{range_pct}%", 
                        str_3m, str_6m, str_12m, 
                        count_90_95, count_95_98, count_98_100
                    ]
                    
                    report_data.append(row)
                    print(row_format.format(*[str(x) for x in row]))

            except Exception as e:
                print(f"Error processing {symbol}: {e}")

        # 6. Export to CSV
        csv_filename = "nrb_local_duration_report.csv"
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(report_data)

        print("-" * 130)
        print(f"Done! Report saved to {os.path.abspath(csv_filename)}")