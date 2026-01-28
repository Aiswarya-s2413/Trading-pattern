import pandas as pd
import copy
import os
from datetime import datetime
from django.core.management.base import BaseCommand
from marketdata.models import Symbol, EodPrice, Parameter
from core.pattern_recognition import get_pattern_triggers

class Command(BaseCommand):
    help = "Run Whipsaw Analysis: Counts Success vs Failed Whipsaws."

    def add_arguments(self, parser):
        parser.add_argument('--symbols', type=str, help='Optional: Comma-separated list of symbols')

    def handle(self, *args, **options):
        # 1. Define Ranges (4 to 10 weeks)
        d1_range = range(4, 11)
        d2_range = range(4, 11)
        
        # 2. Get Symbols
        if options['symbols']:
            symbols = [s.strip() for s in options['symbols'].split(',')]
            self.stdout.write(f"Running for specific symbols: {symbols}")
        else:
            self.stdout.write("Fetching ALL symbols (excluding _BSE)...")
            symbols = Symbol.objects.filter(eodprice__isnull=False)\
                .exclude(symbol__endswith='_BSE')\
                .distinct()\
                .values_list('symbol', flat=True)

        results = []
        total_symbols = len(symbols)
        
        # 🟢 CONFIGURATION (Fixed Defaults)
        PARAM_WEEKS = 52            
        PARAM_COOLDOWN = 52         
        PARAM_DIP = 1.00   # 100% dip threshold to allow deep dips         
        PARAM_LOOKBACK = 52         
        PARAM_SERIES = "rsc30"      

        self.stdout.write(f"Starting analysis on {total_symbols} stocks...")
        self.stdout.write(f"Params: Weeks={PARAM_WEEKS}, Cooldown={PARAM_COOLDOWN}, Series={PARAM_SERIES}")
        self.stdout.write(f"Grid Scan: D1 [4-10] x D2 [4-10]")

        processed_count = 0
        
        for scrip in symbols:
            processed_count += 1
            if processed_count % 10 == 0:
                self.stdout.write(f"Processing... {processed_count}/{total_symbols}")

            try:
                # STEP A: Get Base Breakouts
                base_triggers = get_pattern_triggers(
                    scrip=scrip,
                    pattern="Narrow Range Break",
                    nrb_lookback=PARAM_LOOKBACK,  
                    success_rate=0,
                    weeks=PARAM_WEEKS,
                    series=PARAM_SERIES,
                    cooldown_weeks=PARAM_COOLDOWN, 
                    dip_threshold_pct=PARAM_DIP,   
                    whipsaw_d1=None, 
                    whipsaw_d2=None
                )
                
                if not base_triggers:
                    continue

                earliest_ts = min(t['time'] for t in base_triggers)
                start_date = datetime.fromtimestamp(earliest_ts).date()

                # STEP B: Fetch Data & Build Lookup List
                # We convert QuerySet to a list of dicts {'ts': int, 'val': float} for speed
                processed_data = [] 

                if PARAM_SERIES == 'rsc30':
                    data_qs = Parameter.objects.filter(
                        symbol__symbol=scrip, 
                        trade_date__gte=start_date
                    ).exclude(rsc_sensex_ratio__isnull=True).order_by("trade_date")
                    
                    for row in data_qs:
                        processed_data.append({
                            'ts': int(datetime.combine(row.trade_date, datetime.min.time()).timestamp()),
                            'val': float(row.rsc_sensex_ratio)
                        })
                else:
                    data_qs = EodPrice.objects.filter(
                        symbol__symbol=scrip, 
                        trade_date__gte=start_date
                    ).order_by("trade_date")
                    
                    for row in data_qs:
                        processed_data.append({
                            'ts': int(datetime.combine(row.trade_date, datetime.min.time()).timestamp()),
                            'val': float(row.close)
                        })

                if not processed_data:
                    continue

                # STEP C: Iterate Grid (Manual Detection of Success vs Failure)
                WEEK_SECONDS = 7 * 24 * 60 * 60

                for d1 in d1_range:
                    for d2 in d2_range:
                        d1_seconds = d1 * WEEK_SECONDS
                        d2_seconds = d2 * WEEK_SECONDS

                        success_count = 0
                        failed_count = 0

                        for trigger in base_triggers:
                            breakout_ts = trigger['time']
                            
                            # 1. Find Baseline Value at Breakout Time
                            baseline_val = None
                            start_idx = -1
                            
                            for i, p in enumerate(processed_data):
                                if p['ts'] >= breakout_ts:
                                    baseline_val = p['val']
                                    start_idx = i
                                    break
                            
                            if baseline_val is None: continue

                            # Targets
                            drop_target = baseline_val * 0.90    # -10% Drop
                            recover_target = baseline_val * 1.05 # +5% Recovery
                            d1_deadline = breakout_ts + d1_seconds
                            
                            # 2. Check for Drop (Phase 1)
                            drop_found = False
                            drop_idx = -1
                            drop_ts = 0

                            # Scan forward from breakout to find the drop
                            for k in range(start_idx, len(processed_data)):
                                p = processed_data[k]
                                if p['ts'] > d1_deadline: break # Time limit exceeded
                                
                                if p['val'] <= drop_target:
                                    drop_found = True
                                    drop_idx = k
                                    drop_ts = p['ts']
                                    break
                            
                            if drop_found:
                                # 3. Check for Recovery (Phase 2) - Only if Drop happened
                                d2_deadline = drop_ts + d2_seconds
                                recovery_found = False

                                # Scan forward from the DROP time to find recovery
                                for k in range(drop_idx + 1, len(processed_data)):
                                    p = processed_data[k]
                                    if p['ts'] > d2_deadline: break # Time limit exceeded
                                    
                                    if p['val'] >= recover_target:
                                        recovery_found = True
                                        break
                                
                                if recovery_found:
                                    success_count += 1  # ✅ V-Shape (Success)
                                else:
                                    failed_count += 1   # ❌ Crash (Failure)

                        total_candidates = success_count + failed_count
                        
                        if total_candidates > 0:
                            results.append({
                                'Symbol': scrip,
                                'D1': d1,
                                'D2': d2,
                                'Success_Count': success_count,
                                'Failed_Count': failed_count,
                                'Total_Candidates': total_candidates,
                                'Success_Rate_Pct': (success_count / total_candidates) * 100
                            })

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error on {scrip}: {e}"))

        # 3. Export to Excel
        self.stdout.write("\nGenerating Excel report...")
        
        if results:
            df = pd.DataFrame(results)
            
            # Sheet 1: Global Totals (Summing counts across all stocks for each D1/D2)
            totals_df = df.groupby(['D1', 'D2']).agg({
                'Success_Count': 'sum',
                'Failed_Count': 'sum',
                'Total_Candidates': 'sum'
            }).reset_index()
            totals_df['Global_Success_Rate'] = (totals_df['Success_Count'] / totals_df['Total_Candidates']) * 100
            
            # Sheet 2: Matrix View (Success Rate)
            matrix_success = totals_df.pivot(index='D1', columns='D2', values='Global_Success_Rate').fillna(0)

            # Sheet 3: Stock Matrix (Detailed Counts)
            df['Grouping'] = df.apply(lambda row: f"D1:{row['D1']}_D2:{row['D2']}", axis=1)
            stock_matrix = df.pivot_table(index='Symbol', columns='Grouping', values='Success_Count', aggfunc='sum', fill_value=0)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Whipsaw_Success_Vs_Fail_{timestamp}.xlsx"
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                totals_df.to_excel(writer, sheet_name='Global Totals', index=False)
                matrix_success.to_excel(writer, sheet_name='Matrix_SuccessRate')
                stock_matrix.to_excel(writer, sheet_name='Stock_Success_Counts')
                df.to_excel(writer, sheet_name='Detailed Data', index=False)
            
            self.stdout.write(self.style.SUCCESS(f"✅ Report saved: {os.path.abspath(filename)}"))
        else:
            self.stdout.write(self.style.WARNING("No whipsaws found."))