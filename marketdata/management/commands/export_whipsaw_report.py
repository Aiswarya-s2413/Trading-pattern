import pandas as pd
from datetime import datetime
from django.core.management.base import BaseCommand
from tqdm import tqdm

from marketdata.models import Symbol
from core.pattern_recognition import get_pattern_triggers

class Command(BaseCommand):
    help = 'Generates an Excel report of Whipsaw events (10%, 15%, 20% drops) after NRB breakouts'

    def handle(self, *args, **kwargs):
        self.stdout.write("Starting Whipsaw Report Generation...")

        # 1. CONFIGURATION
        CONFIG = {
            'pattern': "Narrow Range Break",
            'weeks': 52,             
            'cooldown_weeks': 52,     
            'series': "rsc30",       # Or 'close' if you want price breakouts
            'nrb_lookback': 52,
            'success_rate': 0,
            'dip_threshold_pct': 1.00 # Needed to enable the logic in get_pattern_triggers
        }

        # Fetch distinct symbols
        unique_scrips = Symbol.objects.filter(parameter__isnull=False).values_list('symbol', flat=True).distinct()
        total_symbols = len(unique_scrips)
        
        self.stdout.write(f"Scanning {total_symbols} stocks for Whipsaw events...")

        all_whipsaws = []

        # 2. ITERATE OVER STOCKS
        for scrip in tqdm(unique_scrips, total=total_symbols, desc="Processing"):
            
            symbol_obj = Symbol.objects.filter(symbol=scrip).select_related('sector').first()
            if not symbol_obj:
                continue

            # Get Sector Name
            sector_name = "Unknown"
            if symbol_obj.sector:
                sector_name = symbol_obj.sector.name

            try:
                # Call the core logic that now includes _detect_post_breakout_whipsaws
                triggers = get_pattern_triggers(
                    scrip=scrip,
                    pattern=CONFIG['pattern'],
                    nrb_lookback=CONFIG['nrb_lookback'],
                    success_rate=CONFIG['success_rate'],
                    weeks=CONFIG['weeks'],
                    series=CONFIG['series'],
                    cooldown_weeks=CONFIG['cooldown_weeks'],
                    dip_threshold_pct=CONFIG['dip_threshold_pct']
                )

                if not triggers:
                    continue

                # 3. EXTRACT WHIPSAWS
                for t in triggers:
                    # We only care if the trigger actually HAS whipsaws
                    whipsaws = t.get('whipsaws', [])
                    if not whipsaws:
                        continue
                    
                    # Basic Breakout Info
                    breakout_ts = t.get('time')
                    breakout_date = datetime.fromtimestamp(breakout_ts).date()
                    breakout_date_str = breakout_date.strftime('%Y-%m-%d')
                    
                    # Iterate through the whipsaw events for this specific breakout
                    for w in whipsaws:
                        w_ts = w.get('time')
                        w_date = datetime.fromtimestamp(w_ts).date()
                        w_date_str = w_date.strftime('%Y-%m-%d')
                        
                        level_num = w.get('level')
                        level_pct = "Unknown"
                        if level_num == 1: level_pct = "-10%"
                        elif level_num == 2: level_pct = "-15%"
                        elif level_num == 3: level_pct = "-20%"

                        all_whipsaws.append({
                            "Symbol": scrip,
                            "Company": symbol_obj.company_name,
                            "Sector": sector_name,
                            "Breakout Date": breakout_date_str,
                            "Whipsaw Date": w_date_str,
                            "Days After Breakout": (w_date - breakout_date).days,
                            "Whipsaw Level": level_pct,
                            "Whipsaw Price": w.get('price'),
                            "Peak Price Before Drop": w.get('peak_price'),
                            "Drawdown %": w.get('drawdown_pct')
                        })

            except Exception as e:
                continue

        # 4. EXPORT TO EXCEL
        if all_whipsaws:
            df = pd.DataFrame(all_whipsaws)
            
            # Sort by Symbol, then Breakout Date, then Whipsaw Level
            df = df.sort_values(by=['Symbol', 'Breakout Date', 'Drawdown %'])
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            filename = f"Whipsaw_Report_{timestamp}.xlsx"
            
            self.stdout.write(f"Saving {len(df)} events to {filename}...")
            df.to_excel(filename, index=False)
            self.stdout.write(self.style.SUCCESS("Success!"))
        else:
            self.stdout.write(self.style.WARNING("No whipsaw events found."))