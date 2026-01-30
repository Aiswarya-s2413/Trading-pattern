import math
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
import numpy as np
from django.core.management.base import BaseCommand
from django.db import connection, transaction
from django.conf import settings
from marketdata.models import Symbol, Parameter, Index, IndexPrice

USE_POSTGRES = 'postgresql' in settings.DATABASES['default']['ENGINE']

def to_decimal(val):
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return None
    return Decimal(str(val)).quantize(Decimal("1.000000"), rounding=ROUND_HALF_UP)

class Command(BaseCommand):
    help = "Compute RSC NSE (Stock Price / Nifty 50 Price)"

    def handle(self, *args, **options):
        self.stdout.write("\n=== RSC NSE CALCULATION STARTED ===")

        # 1. Load NIFTY 50 Benchmark Data
        try:
            benchmark_qs = IndexPrice.objects.filter(index__symbol="NIFTY 50").order_by("trade_date").values("trade_date", "close")
            benchmark_df = pd.DataFrame.from_records(list(benchmark_qs))
            if benchmark_df.empty: raise ValueError("Empty Data")
            
            benchmark_df['trade_date'] = pd.to_datetime(benchmark_df['trade_date'])
            benchmark_df.set_index('trade_date', inplace=True)
            benchmark_df['nifty_close'] = benchmark_df['close'].astype(float)
            benchmark_df.drop('close', axis=1, inplace=True)
        except Exception:
            self.stderr.write("ERROR: NIFTY 50 data missing. Did you run 'import_nifty'?")
            return

        total_symbols = Symbol.objects.count()
        processed = 0

        # 2. Process Each Stock
        for symbol in Symbol.objects.iterator():
            processed += 1
            if processed % 50 == 0: self.stdout.write(f"Processing... [{processed}/{total_symbols}]")

            # Fetch Stock Prices (Closing Price from Parameter table)
            qs = Parameter.objects.filter(symbol=symbol).order_by("trade_date").values("trade_date", "closing_price")
            df = pd.DataFrame.from_records(list(qs))
            if df.empty: continue

            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df.set_index('trade_date', inplace=True)
            df['stock_close'] = df['closing_price'].astype(float)

            # Join with Nifty
            df = df.join(benchmark_df, how='left')
            df['nifty_close'] = df['nifty_close'].ffill() # Fill holidays

            # --- MATH: Ratio & EMAs ---
            df['rsc_nse_ratio'] = df['stock_close'] / df['nifty_close']
            df['rsc_nse_ema5'] = df['rsc_nse_ratio'].ewm(span=5, adjust=False).mean()
            df['rsc_nse_ema10'] = df['rsc_nse_ratio'].ewm(span=10, adjust=False).mean()

            # Clean Data
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Prepare Update
            rows_to_update = []
            for date, row in df.iterrows():
                if pd.notna(row['rsc_nse_ratio']):
                    rows_to_update.append((
                        date.date(),
                        symbol.id,
                        to_decimal(row['rsc_nse_ratio']),
                        to_decimal(row['rsc_nse_ema5']),
                        to_decimal(row['rsc_nse_ema10'])
                    ))

            # Bulk Update DB
            if rows_to_update:
                self.bulk_update_db(rows_to_update)

        self.stdout.write(self.style.SUCCESS("\n=== CALCULATION COMPLETE ==="))

    def bulk_update_db(self, rows):
        """Efficiently updates DB using SQL"""
        table = Parameter._meta.db_table
        if USE_POSTGRES:
            from psycopg2.extras import execute_values
            sql = f"""
                UPDATE {table} AS p
                SET rsc_nse_ratio = v.ratio, rsc_nse_ema5 = v.ema5, rsc_nse_ema10 = v.ema10
                FROM (VALUES %s) AS v(date, sid, ratio, ema5, ema10)
                WHERE p.trade_date = v.date AND p.symbol_id = v.sid;
            """
            with connection.cursor() as cur:
                execute_values(cur, sql, rows, template="(%s, %s, %s, %s, %s)")
        else:
            # Fallback for SQLite
            with transaction.atomic():
                for r in rows:
                    Parameter.objects.filter(trade_date=r[0], symbol_id=r[1]).update(
                        rsc_nse_ratio=r[2], rsc_nse_ema5=r[3], rsc_nse_ema10=r[4]
                    )