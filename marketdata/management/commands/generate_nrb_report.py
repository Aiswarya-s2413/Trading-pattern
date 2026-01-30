import pandas as pd
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from tqdm import tqdm

from marketdata.models import Symbol, EodPrice
from core.pattern_recognition import get_pattern_triggers

class Command(BaseCommand):
    help = 'Generates Huge NRB Matrix: Weeks (20-104) x Cooldown (20-104)'

    def handle(self, *args, **kwargs):
        self.stdout.write("Starting Huge Matrix NRB Report...")
        self.stdout.write("Iterating Weeks [20-104] AND Cooldowns [20-104]...")

        # 1. CONSTANTS
        SECONDS_IN_A_WEEK = 604800.0
        
        # Base config (Dynamic values will override this)
        CONFIG = {
            'pattern': "Narrow Range Break",
            'series': "rsc30",       
            'success_rate': 0,
        }

        # 2. PRE-FETCH DATA
        self.stdout.write("Pre-fetching Symbol and Sector data...")
        symbol_map = {}
        all_symbols = Symbol.objects.select_related('sector').all()
        for s in all_symbols:
            sector_name = s.sector.name if s.sector else "Unknown"
            symbol_map[s.symbol] = {
                'company': s.company_name,
                'sector': sector_name
            }

        # Filter out stocks ending with '_BSE'
        unique_scrips = Symbol.objects.filter(parameter__isnull=False)\
                                      .exclude(symbol__endswith='_BSE')\
                                      .values_list('symbol', flat=True).distinct()
        
        self.stdout.write(f"Found {len(unique_scrips)} stocks (excluding BSE).")
        
        all_results = []

        # 3. ANALYZE
        for scrip in tqdm(unique_scrips, desc="Analyzing Stocks"):
            if scrip not in symbol_map: continue
            sym_info = symbol_map[scrip]

            # ⚡ OPTIMIZATION: Fetch Price Data ONCE per stock
            price_qs = EodPrice.objects.filter(
                symbol__symbol=scrip,
                close__isnull=False 
            ).values('trade_date', 'close').order_by('trade_date')
            
            if not price_qs: continue

            price_map = {p['trade_date']: float(p['close']) for p in price_qs}
            sorted_dates = sorted(price_map.keys())

            # 🟢 LOOP 1: Weeks (Lookback) from 20 to 104
            for n_weeks in range(20, 105):
                
                # 🟢 LOOP 2: Cooldown from 20 to 104
                for cd_weeks in range(20, 105):
                    try:
                        # --- A. Get Triggers (Double Dynamic Loops) ---
                        triggers = get_pattern_triggers(
                            scrip=scrip,
                            pattern=CONFIG['pattern'],
                            nrb_lookback=n_weeks,    # <--- Matches Weeks
                            weeks=n_weeks,           # <--- Dynamic Weeks
                            success_rate=CONFIG['success_rate'],
                            series=CONFIG['series'],
                            cooldown_weeks=cd_weeks, # <--- Dynamic Cooldown
                            
                            # Disable Dip Check for history
                            dip_threshold_pct=100.0 
                        )

                        if not triggers: continue

                        # --- B. GROUP AGGREGATION ---
                        groups_by_id = {}
                        
                        for t in triggers:
                            gid = t.get('nrb_group_id')
                            if not gid:
                                gid = f"standalone_{t.get('nrb_id')}"
                            
                            if gid not in groups_by_id:
                                groups_by_id[gid] = {
                                    'group_level': t.get('group_level') or t.get('range_high'),
                                    'group_start_time': t.get('group_start_time') or t.get('range_start_time'),
                                    'group_end_time': t.get('group_end_time') or t.get('time'), 
                                    'nrb_count': t.get('group_nrb_count', 1)
                                }

                        # --- C. PROCESS GROUPS ---
                        for gid, group in groups_by_id.items():
                            
                            g_start = group['group_start_time']
                            g_end = group['group_end_time']
                            
                            if g_start and g_end:
                                duration_val = (g_end - g_start) / SECONDS_IN_A_WEEK
                                duration_weeks = round(duration_val, 1)
                            else:
                                duration_weeks = 0

                            if not g_end: continue

                            breakout_dt = datetime.fromtimestamp(g_end).date()
                            start_dt_str = datetime.fromtimestamp(g_start).strftime('%Y-%m-%d') if g_start else "-"
                            breakout_dt_str = breakout_dt.strftime('%Y-%m-%d')

                            # Get Price
                            breakout_price = price_map.get(breakout_dt)
                            if not breakout_price:
                                future_dates = [d for d in sorted_dates if d >= breakout_dt]
                                if future_dates:
                                    breakout_price = price_map[future_dates[0]]
                                    breakout_dt = future_dates[0]
                            
                            if not breakout_price: continue

                            # Returns Calculation
                            def get_pct_change(start_date, start_val, days):
                                target_date = start_date + timedelta(days=days)
                                future_val = next((price_map[d] for d in sorted_dates if d >= target_date), None)
                                if future_val:
                                    return round(((future_val - start_val) / start_val) * 100, 2)
                                return None

                            succ_3m = get_pct_change(breakout_dt, breakout_price, 90)
                            succ_6m = get_pct_change(breakout_dt, breakout_price, 180)
                            succ_9m = get_pct_change(breakout_dt, breakout_price, 270)
                            succ_12m = get_pct_change(breakout_dt, breakout_price, 365)

                            all_results.append({
                                "Symbol": scrip,
                                "Company": sym_info['company'],
                                "Sector": sym_info['sector'],
                                "Weeks Setting": n_weeks,       # <--- New Column
                                "Cooldown Setting": cd_weeks,   # <--- Existing Column
                                "Start Date": start_dt_str,
                                "Breakout Date": breakout_dt_str,
                                "Duration": duration_weeks,
                                "Breakout Level": group['group_level'],
                                "Breakout Price": breakout_price,
                                "3-Month %": succ_3m,
                                "6-Month %": succ_6m,
                                "9-Month %": succ_9m,
                                "12-Month %": succ_12m
                            })

                    except Exception as e:
                        continue

        # 4. EXPORT
        if all_results:
            df = pd.DataFrame(all_results)
            
            # Reorder columns to show Settings first
            cols = ["Symbol", "Company", "Sector", "Weeks Setting", "Cooldown Setting", 
                    "Start Date", "Breakout Date", "Duration", "Breakout Level", 
                    "Breakout Price", "3-Month %", "6-Month %", "9-Month %", "12-Month %"]
            
            final_cols = [c for c in cols if c in df.columns]
            df = df[final_cols]
            
            # Sort order: Weeks -> Cooldown -> Symbol
            df = df.sort_values(by=['Weeks Setting', 'Cooldown Setting', 'Symbol'], ascending=[True, True, True])

            filename = f"NRB_Matrix_Weeks_20-104_Cooldown_20-104_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
            self.stdout.write(f"Saving {len(df)} rows to {filename}...")
            
            # Safety check for Excel row limit (1,048,576 rows)
            if len(df) > 1000000:
                self.stdout.write(self.style.WARNING("⚠️ Warning: Data exceeds 1M rows. Switching to CSV to prevent crash."))
                csv_filename = filename.replace(".xlsx", ".csv")
                df.to_csv(csv_filename, index=False)
                self.stdout.write(self.style.SUCCESS(f"Success! Saved as {csv_filename}"))
            else:
                df.to_excel(filename, index=False)
                self.stdout.write(self.style.SUCCESS(f"Success! Saved as {filename}"))
        else:
            self.stdout.write(self.style.WARNING("No patterns found."))