# marketdata/management/commands/generate_emas.py
import sys
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

from django.core.management.base import BaseCommand
from django.db import connection, transaction
from django.conf import settings
from django.utils import timezone

import pandas as pd
import numpy as np

from marketdata.models import Symbol, EodPrice, Parameter

# CONFIGURATION / TUNABLES
MIN_ROWS_TO_PROCESS = 21         # if < 21 rows, we skip (change if you want)
BATCH_UPSERT_SIZE = 1000        # number of Parameter rows per DB upsert batch
SKIP_SYMBOLS_WITH_LESS_THAN = MIN_ROWS_TO_PROCESS
USE_POSTGRES_UPSERT = 'postgresql' in settings.DATABASES['default']['ENGINE']

# Helper to round Decimal to 4 decimal places for storage
def to_decimal(val, places=4):
    if pd.isna(val):
        return None
    d = Decimal(str(float(val)))
    q = Decimal(10) ** -places
    return d.quantize(q, rounding=ROUND_HALF_UP)

class Command(BaseCommand):
    help = 'Compute EMA21, EMA50, EMA200 for all symbols and store into Parameter table.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--symbols',
            nargs='+',
            help='Optional list of symbol strings to compute (example: "TCS" "INFY")'
        )
        parser.add_argument(
            '--batch-size', type=int, default=BATCH_UPSERT_SIZE,
            help='Number of Parameter rows to upsert per DB batch.'
        )
        parser.add_argument(
            '--skip-small', action='store_true',
            help='Skip symbols with fewer than MIN_ROWS_TO_PROCESS rows (default behavior).'
        )
        parser.add_argument(
            '--incremental', action='store_true',
            help='Only compute for the latest trade_date (incremental mode).'
        )

    def handle(self, *args, **options):
        start_time = timezone.now()
        batch_size = options['batch_size']
        skip_small = options['skip_small']
        incremental = options['incremental']
        symbol_filter = options.get('symbols')

        # Prepare symbol queryset
        qs = Symbol.objects.all().order_by('symbol')
        if symbol_filter:
            qs = qs.filter(symbol__in=symbol_filter)

        total_symbols = qs.count()
        self.stdout.write(f"Starting EMA generation for {total_symbols} symbols. "
                          f"Incremental={incremental}, SkipSmall={skip_small}, BatchSize={batch_size}")

        # We'll process symbol by symbol to keep memory low
        symbol_counter = 0
        for symbol in qs.iterator():
            symbol_counter += 1
            self.stdout.write(f"\n[{symbol_counter}/{total_symbols}] Processing symbol: {symbol.symbol}")
            try:
                self.process_symbol(symbol, batch_size, skip_small, incremental)
            except Exception as e:
                self.stderr.write(f"Error processing {symbol.symbol}: {e}")
                # continue with next symbol
                continue

        elapsed = timezone.now() - start_time
        self.stdout.write(f"\nCompleted. Elapsed: {elapsed}")

    def process_symbol(self, symbol_obj, batch_size, skip_small=True, incremental=False):
        """
        Load EOD data for a single symbol, compute EMAs and upsert into Parameter.
        """

        # Query EodPrice for symbol (ordered by trade_date)
        qs = EodPrice.objects.filter(symbol=symbol_obj).order_by('trade_date').values(
            'trade_date', 'close'
        )

        # Load into DataFrame (pandas handles decimal->float automatically but we convert precisely)
        df = pd.DataFrame.from_records(list(qs))
        if df.empty:
            self.stdout.write(f" - No EOD data for {symbol_obj.symbol}, skipping.")
            return

        # Ensure sorted by date
        df = df.sort_values('trade_date').reset_index(drop=True)

        nrows = len(df)
        self.stdout.write(f" - {nrows} rows available.")

        if skip_small and nrows < SKIP_SYMBOLS_WITH_LESS_THAN:
            self.stdout.write(f" - Skipping {symbol_obj.symbol}: requires >= {SKIP_SYMBOLS_WITH_LESS_THAN} rows.")
            return

        # If incremental mode: process only the latest date(s)
        if incremental:
            last_date = df['trade_date'].max()
            df = df[df['trade_date'] == last_date].copy()
            # For incremental we still need previous EMA values — but we will compute EMAs from full series below and then slice final rows.
            # So set incremental flag but compute using full data; simpler: compute full series and then only upsert last date.
            # To avoid confusion, we'll compute full series but only upsert final dates.
            full_series_needed = True
            df_full = pd.DataFrame.from_records(list(EodPrice.objects.filter(symbol=symbol_obj).order_by('trade_date').values('trade_date', 'close')))
            df_full = df_full.sort_values('trade_date').reset_index(drop=True)
            df_series = df_full
        else:
            df_series = df.copy()

        # For correct EMA, we need the full historical series. So if not incremental we already have df_series as full.
        # If incremental we prepared df_series above.

        # Convert close to float for pandas ewm. We keep original trade_date for indexing.
        df_series['close'] = df_series['close'].astype(float)

        # Compute EMAs using pandas. adjust=False to match trading platforms.
        # Pandas will produce NaN for the first N-1 rows.
        df_series['ema21'] = df_series['close'].ewm(span=21, adjust=False).mean()
        df_series['ema50'] = df_series['close'].ewm(span=50, adjust=False).mean()
        df_series['ema200'] = df_series['close'].ewm(span=200, adjust=False).mean()

        # If incremental -> slice only the last row for upsert
        if incremental:
            df_to_upsert = df_series.tail(1).copy()
        else:
            df_to_upsert = df_series.copy()

        # Prepare upsert rows: convert to list of tuples (trade_date, symbol_id, closing_price, ema21, ema50, ema200)
        rows = []
        for _, r in df_to_upsert.iterrows():
            trade_date = r['trade_date']
            close = to_decimal(r['close'], places=2)
            ema21 = to_decimal(r['ema21'], places=4)
            ema50 = to_decimal(r['ema50'], places=4)
            ema200 = to_decimal(r['ema200'], places=4)

            rows.append({
                'trade_date': trade_date,
                'symbol_id': symbol_obj.id,
                'closing_price': close,
                'ema21': ema21,
                'ema50': ema50,
                'ema200': ema200,
            })

        if not rows:
            self.stdout.write(" - No rows to upsert.")
            return

        # We will use psycopg2.execute_values upsert for Postgres (fast). Otherwise fallback to Django ORM bulk operations.
        if USE_POSTGRES_UPSERT:
            self._postgres_upsert_parameter(rows, batch_size)
        else:
            self._django_bulk_upsert_parameter(rows, batch_size)

    def _postgres_upsert_parameter(self, rows, batch_size):
        """
        Fast Postgres upsert using INSERT ... ON CONFLICT DO UPDATE.
        Expects 'rows' as list of dicts with keys matching Parameter columns.
        """
        import psycopg2
        from psycopg2.extras import execute_values

        table = Parameter._meta.db_table
        columns = ['trade_date', 'symbol_id', 'closing_price', 'ema21', 'ema50', 'ema200']

        # Build SQL
        placeholder = "(" + ",".join(["%s"] * len(columns)) + ")"
        insert_sql = f"""
            INSERT INTO {table} ({', '.join(columns)})
            VALUES %s
            ON CONFLICT (trade_date, symbol_id) DO UPDATE SET
                closing_price = EXCLUDED.closing_price,
                ema21 = EXCLUDED.ema21,
                ema50 = EXCLUDED.ema50,
                ema200 = EXCLUDED.ema200
        """

        # Convert rows into tuples
        tuples = []
        for r in rows:
            tuples.append((
                r['trade_date'],
                r['symbol_id'],
                r['closing_price'],
                r['ema21'],
                r['ema50'],
                r['ema200'],
            ))

        conn = connection.cursor().connection  # psycopg2 connection
        with conn.cursor() as cur:
            for i in range(0, len(tuples), batch_size):
                batch = tuples[i:i + batch_size]
                execute_values(cur, insert_sql, batch, template=placeholder)
            conn.commit()
        self.stdout.write(f" - Upserted {len(tuples)} Parameter rows (Postgres fast path).")

    def _django_bulk_upsert_parameter(self, rows, batch_size):
        """
        Generic fallback using Django ORM. This implementation:
        * tries bulk_create with ignore_conflicts=True (inserts only),
        * then updates existing rows via bulk_update in small batches.
        Note: slower than Postgres upsert, but safe for other DBs.
        """
        created_objects = []
        update_objects = []

        # First, collect existing keys to decide whether to create or update
        keys = [(r['trade_date'], r['symbol_id']) for r in rows]
        trade_dates = [k[0] for k in keys]
        symbol_id = rows[0]['symbol_id'] if rows else None

        existing_qs = Parameter.objects.filter(trade_date__in=trade_dates, symbol_id=symbol_id)
        existing_keys = set((p.trade_date, p.symbol_id) for p in existing_qs)

        for r in rows:
            if (r['trade_date'], r['symbol_id']) in existing_keys:
                p = Parameter(trade_date=r['trade_date'], symbol_id=r['symbol_id'],
                              closing_price=r['closing_price'], ema21=r['ema21'],
                              ema50=r['ema50'], ema200=r['ema200'])
                update_objects.append(p)
            else:
                p = Parameter(trade_date=r['trade_date'], symbol_id=r['symbol_id'],
                              closing_price=r['closing_price'], ema21=r['ema21'],
                              ema50=r['ema50'], ema200=r['ema200'])
                created_objects.append(p)

        # Bulk create (ignores conflicts if DB supports it)
        if created_objects:
            for i in range(0, len(created_objects), batch_size):
                Parameter.objects.bulk_create(created_objects[i:i + batch_size], ignore_conflicts=True)

        # Bulk update existing
        if update_objects:
            # We need to fetch actual DB objects and update fields; below we do an approximate bulk update.
            # Simpler approach: delete existing keys for this symbol & trade_dates then bulk_create all rows.
            # But deletion may be heavy. We'll do per-batch update using a WHERE clause per item (slower).
            for obj in update_objects:
                Parameter.objects.filter(trade_date=obj.trade_date, symbol_id=obj.symbol_id).update(
                    closing_price=obj.closing_price,
                    ema21=obj.ema21,
                    ema50=obj.ema50,
                    ema200=obj.ema200
                )

        self.stdout.write(f" - Created {len(created_objects)}, Updated {len(update_objects)} Parameter rows (Django fallback).")
