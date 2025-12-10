import os
import requests
import csv
from django.conf import settings
from marketdata.models import Symbol, EodPrice
from django.db import transaction

TRUEDATA_BASE_URL = "https://history.truedata.in/getbars"


def retry_failed_eod_import(token, failed_file="failed_bse_symbols.txt", result_file="retry_results.txt"):
    # File path
    file_path = os.path.join(settings.BASE_DIR, "marketdata", "utils",  "data", "logs", failed_file)

    if not os.path.exists(file_path):
        print(f"❌ Failed symbols file not found: {file_path}")
        return

    print(f"\n📌 Retrying EOD import for symbols in: {file_path}")

    # Read symbols
    with open(file_path, "r") as f:
        symbols = [line.strip() for line in f if line.strip()]

    print(f"🔁 Retrying {len(symbols)} symbols...")

    retried_success = []
    retried_failed = []
    missing_fields_log = []

    processed = 0 
    for sym in symbols:
        processed += 1
        print(f"\n🔄 [{processed}/{len(symbols)}] Retrying {sym} ...")

        try:
            symbol_obj = Symbol.objects.filter(symbol=sym).first()
            if not symbol_obj:
                print(f"❌ Symbol not found in DB: {sym}")
                retried_failed.append((sym, "Symbol not in DB"))
                continue

            # Build TrueData URL (last 10 years)
            url = f"{TRUEDATA_BASE_URL}?symbol={sym}&from=140129T09:00:00&to=251208T18:30:00&interval=eod&response=csv"

            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(url, headers=headers)


            if response.status_code != 200:
                print(f"❌ Failed request ({response.status_code}): {sym}")
                retried_failed.append((sym, "HTTP Error"))
                continue

            text = response.text.strip()
            # Symbol not found
            if "symbol does not exist" in text.lower():
                print(f"❌ Symbol not found on TrueData: {sym}")
                retried_failed.append((sym, "Symbol does not exist"))
                continue

            # Unexpected response (HTML, JSON, etc.)
            if text.startswith("<") or text.startswith("{"):
                print(f"❌ Invalid response format for {sym}")
                retried_failed.append((sym, "Invalid response"))
                continue
            # content = response.text.splitlines()

            if len(text.splitlines()) <= 1:
                print(f"⚠️ No EOD data returned: {sym}")
                retried_failed.append((sym, "No Data"))
                continue

            reader = csv.reader(text.splitlines())
            header = next(reader)  # Skip header

            with transaction.atomic():
                for row in reader:
                    try:
                        trade_date = row[0]
                        # Handle missing OHLC values safely
                        def safe_float(val):
                            return float(val) if val not in (None, "", "null", "NULL") else None

                        o = safe_float(row[1])
                        h = safe_float(row[2])
                        l = safe_float(row[3])
                        c = safe_float(row[4])
                        v = safe_float(row[5])

                        # CLOSE is mandatory — skip if missing
                        if c is None:
                            missing_fields_log.append(f"{sym},{trade_date},Missing CLOSE")
                            continue

                        # Log missing open/high/low
                        if o is None or h is None or l is None:
                            missing_fields_log.append(f"{sym},{trade_date},Missing O/H/L")
                    except Exception:
                        print(f"⚠️ Skipping malformed row in {symbol_obj.symbol}: {row}")
                        missing_fields_log.append(f"{sym},{trade_date},Malformed Row")
                        continue

                    EodPrice.objects.update_or_create(
                        trade_date=trade_date,
                        symbol=symbol_obj,
                        defaults={
                            "open": o,
                            "high": h,
                            "low": l,
                            "close": c,
                            "volume": v,
                        }
                    )

            print(f"✅ Successfully retried: {sym}")
            retried_success.append(sym)

        except Exception as e:
            print(f"❌ Error processing {sym}: {e}")
            retried_failed.append((sym, str(e)))

    # Save retry results
    retry_log_path = os.path.join(
        settings.BASE_DIR,
        "marketdata",
        "utils",
        "data",
        "logs",
        result_file
    )

    with open(retry_log_path, "w") as f:
        f.write("=== RETRY RESULTS ===\n\n")
        f.write("Successful:\n")
        for s in retried_success:
            f.write(f"{s}\n")
        f.write("\nFailed Again:\n")
        for s, msg in retried_failed:
            f.write(f"{s} — {msg}\n")

    print(f"\nRetry results saved to: {retry_log_path}")

    return retried_success, retried_failed



def check_unavailable_bse_eod_import(token,  result_file="unavailable_bse_eod_results.txt"):
    # File path
    # file_path = os.path.join(settings.BASE_DIR, "marketdata", "utils",  "data", "logs", failed_file)

    # if not os.path.exists(file_path):
    #     print(f"❌ Failed symbols file not found: {file_path}")
    #     return

    # print(f"\n📌 Retrying EOD import for symbols in: {file_path}")
    from marketdata.utils.data.unavailable_bse_symbols import UNAVAILABLE_BSE as symbols
    # Read symbols
    # with open(file_path, "r") as f:
    #     symbols = [line.strip() for line in f if line.strip()]


    print(f"🔁 Retrying {len(symbols)} symbols...")

    retried_success = []
    retried_failed = []
    processed = 0 
    for s in symbols:
        sym = f"{s.strip()}_BSE"
        processed += 1
        print(f"\n🔄 [{processed}/{len(symbols)}] Retrying {sym} ...")

        try:
            symbol_obj = Symbol.objects.filter(symbol=sym).first()
            if not symbol_obj:
                print(f"❌ Symbol not found in DB: {sym}")
                retried_failed.append((sym, "Symbol not in DB"))

            # Build TrueData URL (last 10 years)
            url = f"{TRUEDATA_BASE_URL}?symbol={sym}&from=140129T09:00:00&to=251208T18:30:00&interval=eod&response=csv"

            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(url, headers=headers)


            if response.status_code != 200:
                print(f"❌ Failed request ({response.status_code}): {sym}")
                retried_failed.append((sym, "HTTP Error"))
                continue

            text = response.text.strip()
            # Symbol not found
            if "symbol does not exist" in text.lower():
                print(f"❌ Symbol not found on TrueData: {sym}")
                retried_failed.append((sym, "Symbol does not exist"))
                continue

            # Unexpected response (HTML, JSON, etc.)
            if text.startswith("<") or text.startswith("{"):
                print(f"❌ Invalid response format for {sym}")
                retried_failed.append((sym, "Invalid response"))
                continue
            # content = response.text.splitlines()

            if len(text.splitlines()) <= 1:
                print(f"⚠️ No EOD data returned: {sym}")
                retried_failed.append((sym, "No Data"))
                continue

            reader = csv.reader(text.splitlines())
            header = next(reader)  # Skip header

            # with transaction.atomic():
            #     for row in reader:
            #         try:
            #             trade_date = row[0]
            #             o = float(row[1])
            #             h = float(row[2])
            #             l = float(row[3])
            #             c = float(row[4])
            #             v = int(float(row[5]))
            #         except Exception:
            #             print(f"⚠️ Skipping malformed row in {symbol_obj.symbol}: {row}")
            #             continue

            #         EodPrice.objects.update_or_create(
            #             trade_date=trade_date,
            #             symbol=symbol_obj,
            #             defaults={
            #                 "open": o,
            #                 "high": h,
            #                 "low": l,
            #                 "close": c,
            #                 "volume": v,
            #             }
            #         )

            print(f"✅ Successfully retried: {sym}")
            retried_success.append(sym)

        except Exception as e:
            print(f"❌ Error processing {sym}: {e}")
            retried_failed.append((sym, str(e)))

    # Save retry results
    retry_log_path = os.path.join(
        settings.BASE_DIR,
        "marketdata",
        "utils",
        "data",
        "logs",
        result_file
    )

    with open(retry_log_path, "w") as f:
        f.write("=== RETRY RESULTS ===\n\n")
        f.write("Successful:\n")
        for s in retried_success:
            f.write(f"{s}\n")
        f.write("\nFailed Again:\n")
        for s, msg in retried_failed:
            f.write(f"{s} — {msg}\n")

    print(f"\nRetry results saved to: {retry_log_path}")

    return retried_success, retried_failed