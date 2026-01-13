import pandas as pd
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from tqdm import tqdm

from marketdata.models import Symbol, EodPrice
from core.pattern_recognition import get_pattern_triggers

class Command(BaseCommand):
    help = 'Generates NRB Group Report (Cyan Lines) without Deep Dip limits'

    def handle(self, *args, **kwargs):
        self.stdout.write("Starting NRB Report (Unlimited Dip)...")

        # 1. CONFIGURATION
        COOLDOWN_WEEKS = 28
        
        SECONDS_IN_A_WEEK = 604800.0
        
        CONFIG = {
            'pattern': "Narrow Range Break",
            'weeks': 52,             
            'cooldown_weeks': COOLDOWN_WEEKS,     
            'series': "rsc30",       
            'nrb_lookback': 52,
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

        unique_scrips = Symbol.objects.filter(parameter__isnull=False).values_list('symbol', flat=True).distinct()
        
        all_results = []

        # 3. ANALYZE
        for scrip in tqdm(unique_scrips, desc="Analyzing Stocks"):
            if scrip not in symbol_map: continue
            sym_info = symbol_map[scrip]

            try:
                # --- A. Get Triggers (Unlimited Dip) ---
                triggers = get_pattern_triggers(
                    scrip=scrip,
                    pattern=CONFIG['pattern'],
                    nrb_lookback=CONFIG['nrb_lookback'],
                    success_rate=CONFIG['success_rate'],
                    weeks=CONFIG['weeks'],
                    series=CONFIG['series'],
                    cooldown_weeks=CONFIG['cooldown_weeks'],
                    
                    # 🔴 DISABLE DIP CHECK by setting it to 10000% (100.0)
                    # This allows the line to extend back fully regardless of price drops.
                    dip_threshold_pct=100.0 
                )

                if not triggers: continue

                # --- B. Get Price Data ---
                price_qs = EodPrice.objects.filter(
                    symbol__symbol=scrip,
                    close__isnull=False 
                ).values('trade_date', 'close').order_by('trade_date')
                
                if not price_qs: continue

                price_map = {p['trade_date']: float(p['close']) for p in price_qs}
                sorted_dates = sorted(price_map.keys())

                # --- C. GROUP AGGREGATION (Cyan Lines) ---
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

                # --- D. PROCESS GROUPS ---
                for gid, group in groups_by_id.items():
                    
                    # 1. Calculate Duration
                    g_start = group['group_start_time']
                    g_end = group['group_end_time']
                    
                    if g_start and g_end:
                        duration_val = (g_end - g_start) / SECONDS_IN_A_WEEK
                        duration_weeks = round(duration_val, 1)
                    else:
                        duration_weeks = 0

                    # 2. Get Dates
                    if not g_end: continue

                    breakout_dt = datetime.fromtimestamp(g_end).date()
                    start_dt_str = datetime.fromtimestamp(g_start).strftime('%Y-%m-%d') if g_start else "-"
                    breakout_dt_str = breakout_dt.strftime('%Y-%m-%d')

                    # 3. Get Price at Breakout
                    breakout_price = price_map.get(breakout_dt)
                    if not breakout_price:
                        future_dates = [d for d in sorted_dates if d >= breakout_dt]
                        if future_dates:
                            breakout_price = price_map[future_dates[0]]
                            breakout_dt = future_dates[0]
                    
                    if not breakout_price: continue

                    # 4. Returns
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

            except Exception:
                continue

        # 4. EXPORT
        if all_results:
            df = pd.DataFrame(all_results)
            cols = ["Symbol", "Company", "Sector", "Start Date", "Breakout Date", 
                    "Duration", "Breakout Level", "Breakout Price", 
                    "3-Month %", "6-Month %", "9-Month %", "12-Month %"]
            
            final_cols = [c for c in cols if c in df.columns]
            df = df[final_cols]
            df = df.sort_values(by=['Symbol', 'Breakout Date'], ascending=[True, False])

            filename = f"NRB_UnlimitedDip_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
            self.stdout.write(f"Saving {len(df)} rows to {filename}...")
            df.to_excel(filename, index=False)
            self.stdout.write(self.style.SUCCESS("Success!"))
        else:
            self.stdout.write(self.style.WARNING("No patterns found."))