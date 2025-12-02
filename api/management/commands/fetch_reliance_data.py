# api/management/commands/fetch_reliance_data.py

from django.core.management.base import BaseCommand
from api.models import PriceFeature
import yfinance as yf
import pandas as pd
from decimal import Decimal

TICKER = "RELIANCE.NS"   # yfinance symbol
YEARS = "10y"            # 10 years


class Command(BaseCommand):
    help = "Fetch 10 years of data for RELIANCE.NS and store in PriceFeature table."

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS(f"Fetching {YEARS} of data for {TICKER} from yfinance..."))

        df = yf.download(TICKER, period=YEARS, interval="1d", auto_adjust=False)

        if df.empty:
            self.stdout.write(self.style.ERROR("No data returned from yfinance."))
            return

        # ─────────────────────────────────────────────
        # 1) Normalize / fix columns from yfinance
        # ─────────────────────────────────────────────
        # Sometimes yfinance returns MultiIndex columns like ('Open', 'RELIANCE.NS')
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(col[0]) for col in df.columns]
        else:
            df.columns = [str(col) for col in df.columns]

        # Standardize common OHLCV names to: Open, High, Low, Close, Volume, Adj Close
        rename_map = {}
        for c in df.columns:
            lc = c.lower().strip()
            if lc in ["open", "high", "low", "close", "volume", "adj close", "adj_close"]:
                if lc in ["adj close", "adj_close"]:
                    rename_map[c] = "Adj Close"
                elif lc == "volume":
                    rename_map[c] = "Volume"
                else:
                    rename_map[c] = lc.capitalize()

        if rename_map:
            df = df.rename(columns=rename_map)

        required_cols = ["Open", "High", "Low", "Close"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            self.stdout.write(self.style.ERROR(f"Missing expected columns: {missing}"))
            self.stdout.write(self.style.ERROR(f"Actual columns: {list(df.columns)}"))
            return

        # Drop rows with missing OHLC values
        df = df.dropna(subset=required_cols)

        # ─────────────────────────────────────────────
        # 2) Feature calculations
        # ─────────────────────────────────────────────
        # EMAs (50, 200)
        df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
        df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

        # Simple placeholder for RSC_20: Close / 20-day rolling mean
        rolling_mean_20 = df["Close"].rolling(window=20).mean()
        df["RSC_20"] = df["Close"] / rolling_mean_20

        # Dummy success metric:
        # If next-day return > 1%, mark as 100, else 0
        df["Next_Close"] = df["Close"].shift(-1)
        df["Return_pct"] = (df["Next_Close"] - df["Close"]) / df["Close"] * 100
        df["is_successful_trade"] = df["Return_pct"].apply(
            lambda x: 100 if pd.notna(x) and x > 1 else 0
        )

        # ─────────────────────────────────────────────
        # 3) Save to DB via PriceFeature
        # ─────────────────────────────────────────────
        created, updated = 0, 0

        for idx, row in df.iterrows():
            # Index from yfinance is usually a Timestamp
            date = idx.date()

            obj, is_created = PriceFeature.objects.update_or_create(
                symbol=TICKER,
                date=date,
                defaults={
                    "open": Decimal(str(round(row["Open"], 2))),
                    "high": Decimal(str(round(row["High"], 2))),
                    "low": Decimal(str(round(row["Low"], 2))),
                    "close": Decimal(str(round(row["Close"], 2))),
                    "EMA_50": (
                        Decimal(str(round(row["EMA_50"], 2)))
                        if not pd.isna(row["EMA_50"])
                        else None
                    ),
                    "EMA_200": (
                        Decimal(str(round(row["EMA_200"], 2)))
                        if not pd.isna(row["EMA_200"])
                        else None
                    ),
                    "RSC_20": (
                        Decimal(str(round(row["RSC_20"], 4)))
                        if not pd.isna(row["RSC_20"])
                        else None
                    ),
                    "is_successful_trade": (
                        Decimal(str(round(row["is_successful_trade"], 2)))
                        if not pd.isna(row["is_successful_trade"])
                        else Decimal("0.00")
                    ),
                }
            )

            if is_created:
                created += 1
            else:
                updated += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Done. Created: {created}, Updated: {updated} rows for {TICKER}."
            )
        )
