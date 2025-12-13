import math
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd
import numpy as np

from django.core.management.base import BaseCommand
from django.db import connection, transaction
from django.conf import settings

from marketdata.models import Symbol, Parameter, Index, IndexPrice

BATCH_SIZE = 2000
USE_POSTGRES = 'postgresql' in settings.DATABASES['default']['ENGINE']


def to_decimal(val, places=6):
    """Convert to Decimal with specified precision"""
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return None
    d = Decimal(str(float(val)))
    q = Decimal(10) ** -places
    return d.quantize(q, rounding=ROUND_HALF_UP)


class Command(BaseCommand):
    help = "Compute RSC SENSEX (Pine Script logic) - daily price ratio with EMAs"

    def add_arguments(self, parser):
        parser.add_argument('--incremental', action='store_true',
                            help='Only update the latest trade_date per symbol (fast).')

    def handle(self, *args, **options):
        incremental = options.get('incremental', False)

        self.stdout.write("\n=== RSC SENSEX CALCULATION STARTED ===")

        # Load SENSEX index
        try:
            self.sensex = Index.objects.get(symbol="SENSEX")
        except Index.DoesNotExist:
            self.stderr.write("ERROR: Index with symbol 'SENSEX' not found.")
            return

        # Preload SENSEX series
        self.sensex_df = self.load_index_series(self.sensex)

        total_symbols = Symbol.objects.count()
        self.stdout.write(f"Found {total_symbols} symbols. incremental={incremental}")

        processed = 0
        total_updated = 0
        total_skipped = 0

        for symbol in Symbol.objects.iterator():
            processed += 1
            self.stdout.write(f"\n[{processed}/{total_symbols}] {symbol.symbol}")

            try:
                updated_count, skipped_count = self.process_symbol(symbol, incremental)
                total_updated += updated_count
                total_skipped += skipped_count
            except Exception as e:
                self.stderr.write(f"Error for {symbol.symbol}: {e}")
                import traceback
                self.stderr.write(traceback.format_exc())
                continue

        self.stdout.write("\n=== RSC SENSEX CALCULATION COMPLETED ===")
        self.stdout.write(f"Total symbols processed: {processed}. Rows updated: {total_updated}. Rows skipped: {total_skipped}.\n")

    def load_index_series(self, index_obj):
        """Load index price series as DataFrame"""
        qs = IndexPrice.objects.filter(index=index_obj).order_by("trade_date").values("trade_date", "close")
        df = pd.DataFrame.from_records(list(qs))
        if df.empty:
            raise ValueError(f"No IndexPrice data for {index_obj.symbol}")

        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        df['sensex_close'] = df['close'].astype(float)
        df.drop('close', axis=1, inplace=True)
        return df

    def process_symbol(self, symbol, incremental=False):
        """
        Compute RSC SENSEX ratio and EMAs for a symbol
        
        Pine Script Logic:
        as = security(a, timeFrame, close)  -> stock_close
        bs = security(b, timeFrame, close)  -> sensex_close
        ratio = as/bs
        ema5 = ema(ratio, 5)
        ema10 = ema(ratio, 10)
        """
        # Load parameter series (closing_price) for this symbol
        qs = Parameter.objects.filter(symbol=symbol).order_by("trade_date").values("trade_date", "closing_price")
        df = pd.DataFrame.from_records(list(qs))
        if df.empty:
            return 0, 0

        # Normalize
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        df['stock_close'] = df['closing_price'].astype(float)

        # Join SENSEX series (left join to keep all stock dates)
        df = df.join(self.sensex_df, how='left')

        # Forward fill SENSEX prices for weekends/holidays
        df['sensex_close'] = df['sensex_close'].ffill()

        # ===== COMPUTE RSC RATIO (as/bs) =====
        df['rsc_sensex_ratio'] = df['stock_close'] / df['sensex_close']

        # Replace inf/nan with None
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # ===== COMPUTE EMAs =====
        # EMA with span=5 matches Pine Script ema(source, 5)
        df['rsc_sensex_ema5'] = df['rsc_sensex_ratio'].ewm(span=5, adjust=False).mean()
        df['rsc_sensex_ema10'] = df['rsc_sensex_ratio'].ewm(span=10, adjust=False).mean()

        # If incremental: only keep the last row
        if incremental:
            df_to_upsert = df.tail(1)
        else:
            df_to_upsert = df

        # Prepare rows for batch update
        rows = []
        skipped = 0
        
        for trade_date, row in df_to_upsert.iterrows():
            ratio = row.get('rsc_sensex_ratio', None)
            ema5 = row.get('rsc_sensex_ema5', None)
            ema10 = row.get('rsc_sensex_ema10', None)

            # Convert NaN to None
            if pd.isna(ratio):
                ratio = None
            if pd.isna(ema5):
                ema5 = None
            if pd.isna(ema10):
                ema10 = None

            # Skip if all values are None
            if all(v is None for v in [ratio, ema5, ema10]):
                skipped += 1
                continue

            # Convert to date for psycopg2 compatibility
            td = trade_date.date() if hasattr(trade_date, 'date') else trade_date

            rows.append((
                td, 
                symbol.id,
                to_decimal(ratio, 6),
                to_decimal(ema5, 6),
                to_decimal(ema10, 6)
            ))

        if rows:
            self.upsert_rsc(rows)

        return len(rows), skipped

    def upsert_rsc(self, rows):
        """
        Batch-upsert rows using Postgres UPDATE FROM VALUES.
        rows: list of tuples (trade_date, symbol_id, rsc_sensex_ratio, rsc_sensex_ema5, rsc_sensex_ema10)
        """
        if not rows:
            return

        table = Parameter._meta.db_table
        
        if USE_POSTGRES:
            from psycopg2.extras import execute_values

            sql = f"""
                UPDATE {table} AS p
                SET rsc_sensex_ratio = v.rsc_sensex_ratio::numeric,
                    rsc_sensex_ema5 = v.rsc_sensex_ema5::numeric,
                    rsc_sensex_ema10 = v.rsc_sensex_ema10::numeric
                FROM (VALUES %s) AS v(trade_date, symbol_id, rsc_sensex_ratio, rsc_sensex_ema5, rsc_sensex_ema10)
                WHERE p.trade_date = v.trade_date
                  AND p.symbol_id = v.symbol_id;
            """

            total = 0
            with connection.cursor() as cur, transaction.atomic():
                for i in range(0, len(rows), BATCH_SIZE):
                    batch = rows[i:i + BATCH_SIZE]
                    execute_values(cur, sql, batch, template="(%s,%s,%s,%s,%s)")
                    total += len(batch)

            self.stdout.write(f" - Updated {total} RSC SENSEX rows.")
        else:
            # Fallback for non-Postgres databases
            with transaction.atomic():
                for row in rows:
                    Parameter.objects.filter(
                        trade_date=row[0],
                        symbol_id=row[1]
                    ).update(
                        rsc_sensex_ratio=row[2],
                        rsc_sensex_ema5=row[3],
                        rsc_sensex_ema10=row[4]
                    )
            self.stdout.write(f" - Updated {len(rows)} RSC SENSEX rows.")