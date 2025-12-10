import math
from datetime import timedelta
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd
import numpy as np

from django.core.management.base import BaseCommand
from django.db import connection, transaction
from django.conf import settings

from marketdata.models import Symbol, Parameter, Index, IndexPrice

N_DAYS = 50
BATCH_SIZE = 2000  # execute_values chunk size
USE_POSTGRES = 'postgresql' in settings.DATABASES['default']['ENGINE']


def to_decimal(val, places=4):
    if val is None or (isinstance(val, float) and (math.isnan(val))):
        return None
    d = Decimal(str(float(val)))
    q = Decimal(10) ** -places
    return d.quantize(q, rounding=ROUND_HALF_UP)


class Command(BaseCommand):
    help = "Compute RSC30 and RSC500 for all symbols."

    def add_arguments(self, parser):
        parser.add_argument('--incremental', action='store_true',
                            help='Only update the latest trade_date per symbol (fast).')

    def handle(self, *args, **options):
        incremental = options.get('incremental', False)

        self.stdout.write("\n=== RSC CALCULATION STARTED ===")

        # Load index master objects
        try:
            self.sensex = Index.objects.get(symbol="SENSEX")
        except Index.DoesNotExist:
            self.stderr.write("ERROR: Index with symbol 'SENSEX' not found.")
            return
        try:
            self.nifty500 = Index.objects.get(symbol="NIFTY500")
        except Index.DoesNotExist:
            self.stderr.write("ERROR: Index with symbol 'NIFTY500' not found.")
            return

        # Preload index series (and normalize)
        self.sensex_df = self.load_index_series(self.sensex).rename(columns={"close": "sensex_close"})
        self.nifty500_df = self.load_index_series(self.nifty500).rename(columns={"close": "n500_close"})

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
                continue

        self.stdout.write("\n=== RSC CALCULATION COMPLETED ===")
        self.stdout.write(f"Total symbols processed: {processed}. Rows updated: {total_updated}. Rows skipped: {total_skipped}.\n")

    def load_index_series(self, index_obj):
        qs = IndexPrice.objects.filter(index=index_obj).order_by("trade_date").values("trade_date", "close")
        df = pd.DataFrame.from_records(list(qs))
        if df.empty:
            raise ValueError(f"No IndexPrice data for {index_obj.symbol}")

        # normalize index type to pandas datetime index
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        df['close'] = df['close'].astype(float)
        return df

    def process_symbol(self, symbol, incremental=False):
        # Load parameter series (closing_price) for this symbol
        qs = Parameter.objects.filter(symbol=symbol).order_by("trade_date").values("trade_date", "closing_price")
        df = pd.DataFrame.from_records(list(qs))
        if df.empty:
            # nothing to compute
            return 0, 0

        # normalize
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        df['closing_price'] = df['closing_price'].astype(float)

        # map to expected column name for computations
        df['stock_close'] = df['closing_price']

        # If incremental: we only need to upsert last date for each symbol,
        # but to compute correctly we need the historical series for shift.
        # We'll compute full series and then take last row.
        # (Alternatively, you can store previous shift values in DB to avoid full recompute.)
        # Join index series (index dfs already datetime-indexed)
        # The index series might have different calendar (holidays) — join by date (left)
        df = df.join(self.sensex_df, how='left')
        df = df.join(self.nifty500_df, how='left')

        # Compute returns safely
        # stock_ret = stock_close / stock_close.shift(N_DAYS)
        df['stock_ret'] = df['stock_close'] / df['stock_close'].shift(N_DAYS)
        df['sensex_ret'] = df['sensex_close'] / df['sensex_close'].shift(N_DAYS)
        df['n500_ret'] = df['n500_close'] / df['n500_close'].shift(N_DAYS)

        # Prevent divide-by-zero / inf issues
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Compute RSC
        df['rsc30'] = df['stock_ret'] / df['sensex_ret']
        df['rsc500'] = df['stock_ret'] / df['n500_ret']

        # If incremental: only keep the last available date row(s)
        if incremental:
            df_to_upsert = df.tail(1)
        else:
            df_to_upsert = df

        rows = []
        skipped = 0
        for trade_date, row in df_to_upsert.iterrows():
            r30 = row.get('rsc30', None)
            r500 = row.get('rsc500', None)

            # Convert NaN to None
            if pd.isna(r30):
                r30 = None
            if pd.isna(r500):
                r500 = None

            if r30 is None and r500 is None:
                skipped += 1
                continue

            # Use Python date for psycopg2 compatibility
            td = trade_date.date() if hasattr(trade_date, 'date') else trade_date

            rows.append((td, symbol.id, to_decimal(r30, 4), to_decimal(r500, 4)))

        if rows:
            self.upsert_rsc(rows)

        return len(rows), skipped

    def upsert_rsc(self, rows):
        """
        Batch-upsert rows using Postgres UPDATE FROM VALUES.
        rows: list of tuples (trade_date(date), symbol_id(int), rsc30(Decimal|None), rsc500(Decimal|None))
        """
        if not rows:
            return

        table = Parameter._meta.db_table
        # We'll chunk rows to avoid too-large SQL payloads
        from psycopg2.extras import execute_values

        sql = f"""
            UPDATE {table} AS p
            SET rsc30 = v.rsc30::numeric,
                rsc500 = v.rsc500::numeric
            FROM (VALUES %s) AS v(trade_date, symbol_id, rsc30, rsc500)
            WHERE p.trade_date = v.trade_date
              AND p.symbol_id = v.symbol_id;

        """

        total = 0
        with connection.cursor() as cur, transaction.atomic():
            for i in range(0, len(rows), BATCH_SIZE):
                batch = rows[i:i + BATCH_SIZE]
                execute_values(cur, sql, batch, template="(%s,%s,%s,%s)")
                total += len(batch)

        self.stdout.write(f" - Updated {total} RSC rows.")