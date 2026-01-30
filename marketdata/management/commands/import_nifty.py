import yfinance as yf
import pandas as pd
from django.core.management.base import BaseCommand
from marketdata.models import Index, IndexPrice

class Command(BaseCommand):
    help = "Import historical Nifty 50 data from Yahoo Finance"

    def handle(self, *args, **options):
        self.stdout.write("--- Starting Nifty 50 Import ---")

        # 1. Create/Get the NIFTY 50 Index bucket
        index_symbol = "NIFTY 50"
        nifty_index, created = Index.objects.get_or_create(
            symbol=index_symbol,
            defaults={"name": "Nifty 50", "exchange": "NSE"}
        )

        if created:
            self.stdout.write(self.style.SUCCESS(f"Created Index: {index_symbol}"))
        else:
            self.stdout.write(f"Found Index: {index_symbol}")

        # 2. Download Data
        self.stdout.write("Downloading data from Yahoo Finance (^NSEI)...")
        # Downloads data from year 2000 to now
        df = yf.download("^NSEI", start="2000-01-01", progress=False)

        if df.empty:
            self.stderr.write("No data downloaded. Check internet connection.")
            return

        # 3. Save to Database
        objects_to_create = []
        # Get existing dates to avoid duplicates
        existing_dates = set(IndexPrice.objects.filter(index=nifty_index).values_list('trade_date', flat=True))
        
        count = 0
        for date, row in df.iterrows():
            trade_date = date.date()
            if trade_date in existing_dates:
                continue

            # Extract scalar values from the Series
            def get_val(series):
                val = series.iloc[0] if isinstance(series, pd.Series) else series
                return float(val) if pd.notna(val) else 0.0

            obj = IndexPrice(
                index=nifty_index,
                trade_date=trade_date,
                open=get_val(row['Open']),
                high=get_val(row['High']),
                low=get_val(row['Low']),
                close=get_val(row['Close']),
                volume=int(get_val(row['Volume']))
            )
            objects_to_create.append(obj)

            # Bulk create in batches of 1000
            if len(objects_to_create) >= 1000:
                IndexPrice.objects.bulk_create(objects_to_create)
                count += len(objects_to_create)
                objects_to_create = []
                self.stdout.write(f"Saved {count} records...")

        # Save remaining
        if objects_to_create:
            IndexPrice.objects.bulk_create(objects_to_create)
            count += len(objects_to_create)

        self.stdout.write(self.style.SUCCESS(f"Completed! Total records added: {count}"))