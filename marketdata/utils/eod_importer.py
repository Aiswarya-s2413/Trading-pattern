import os
import csv
import requests
import time
from datetime import datetime, timedelta
from django.conf import settings
from django.db import transaction
from marketdata.models import Symbol, EodPrice

# -------------------------------------------
# TrueData API Base
# -------------------------------------------
HISTORY_URL = "https://history.truedata.in/getbars"


def get_last_10_year_range():
    """
    Returns TrueData-compatible date range where:
    - 'to' = yesterday (EOD completed)
    - 'from' = yesterday minus 10 years
    Format: YYMMDDT09:00:00 and YYMMDDT18:30:00
    """
    today = datetime.today()
    yesterday = today - timedelta(days=1)

    # end of yesterday trading session
    to_str = yesterday.strftime("%y%m%dT18:30:00")

    # 10 years before yesterday
    ten_years_ago = yesterday.replace(year=yesterday.year - 10)
    from_str = ten_years_ago.strftime("%y%m%dT09:00:00")

    return from_str, to_str


def fetch_eod_data(symbol, token, max_retries=3):
    from_str, to_str = get_last_10_year_range()

    params = {
        "symbol": symbol,
        "from": from_str,
        "to": to_str,
        "interval": "eod",
        "response": "csv"
    }

    headers = {"Authorization": f"Bearer {token}"}

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(HISTORY_URL, params=params, headers=headers, timeout=20)

            if response.status_code == 200:
                text = response.text.strip()

                if text == "":
                    raise Exception("Empty response")
                
                # Symbol not found
                if "symbol does not exist" in text.lower():
                    raise Exception("Symbol does not exist")

                reader = csv.reader(text.splitlines())
                return list(reader)

            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")

        except Exception as e:
            print(f"⚠️ Retry {attempt}/{max_retries} failed for {symbol}: {e}")
            time.sleep(2)

    raise Exception(f"Failed after {max_retries} attempts")


def import_eod_for_all_symbols(token):
    """
    Main ETL function.
    Loops through Symbol table and loads EOD data into EodPrice.
    """
    success_log = []
    error_log = []

    # Paths for storing logs
    base_dir = settings.BASE_DIR
    failed_log_path = os.path.join(base_dir, "marketdata", "utils", "data", "failed_symbols.txt")
    success_log_path = os.path.join(base_dir, "marketdata", "utils", "data", "success_symbols.txt")

    symbols = Symbol.objects.filter(market_type="BSE")  # NSE ONLY
    total = symbols.count()
    processed = 0

    for sym in symbols:
        processed += 1
        print(f"\n🔄 [{processed}/{total}] Processing {sym.symbol} ...")

        try:
            with transaction.atomic():  # Atomic per symbol
                rows = fetch_eod_data(sym.symbol, token)

                if not rows:
                    raise Exception("No rows in CSV")

                for row in rows:
                    # Skip header
                    if row[0].lower() == "date" or row[1].lower() == "open":
                        continue

                    try:
                        trade_date = row[0]
                        o = float(row[1])
                        h = float(row[2])
                        l = float(row[3])
                        c = float(row[4])
                        v = int(float(row[5]))
                    except Exception:
                        print(f"⚠️ Skipping malformed row in {sym.symbol}: {row}")
                        continue

                    EodPrice.objects.update_or_create(
                        trade_date=trade_date,
                        symbol=sym,
                        defaults={
                            "open": o,
                            "high": h,
                            "low": l,
                            "close": c,
                            "volume": v,
                        }
                    )

                success_log.append(sym.symbol)
                print(f"✅ SUCCESS: Stored EOD for {sym.symbol}")

        except Exception as e:
            print(f"❌ FAILED: {sym.symbol} — {e}")
            error_log.append(f"{sym.symbol} — {str(e)}")

    # Save logs
    with open(failed_log_path, "w", encoding="utf-8") as f:
        for line in error_log:
            f.write(line + "\n")

    with open(success_log_path, "w", encoding="utf-8") as f:
        for line in success_log:
            f.write(line + "\n")

    print("\n📌 SUMMARY")
    print("Success:", success_log)
    print("Failed:", error_log)
    print(f"\n📝 Logs saved to:\n - {success_log_path}\n - {failed_log_path}")

    return success_log, error_log
