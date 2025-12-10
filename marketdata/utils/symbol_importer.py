import csv
import os
from django.conf import settings
from marketdata.models import Symbol, Sectors
from marketdata.utils.data.available_bse_symbols import AVAILABLE_BSE
from datetime import datetime   

CSV_COLUMNS = {
    "id": 0,             # for reporting only
    "symbol": 1,         # SBIN, TCS, etc.
    "sector": 2,         # Abrasives, Agriculture, etc.
    "company_name": 5,   # Company Name
}

def import_symbols_from_csv(csv_name="Analyst_Workday.csv"):
    # Build full path
    csv_path = os.path.join(
        settings.BASE_DIR,
        "marketdata",
        "utils",
        "data",
        csv_name
    )

    print(f"\n📄 Using CSV file: {csv_path}\n")

    added = []
    skipped = []

    with open(csv_path, newline='', encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            row_id = row[CSV_COLUMNS["id"]].strip()
            symbol = row[CSV_COLUMNS["symbol"]].strip()
            sector_name = row[CSV_COLUMNS["sector"]].strip()
            company = row[CSV_COLUMNS["company_name"]].strip()

            # Skip if any required field is missing
            if not symbol or not sector_name or not company:
                skipped.append((row_id, symbol))
                print(f"⛔ Skipped row {row_id}: Missing required fields")
                continue

            # Ensure Sector exists
            sector_obj, _ = Sectors.objects.get_or_create(name=sector_name)

            symbol = f"{symbol}_BSE"

            # Create Symbol
            obj, created = Symbol.objects.get_or_create(
                symbol=symbol,
                defaults={
                    "company_name": company,
                    "sector": sector_obj,
                    "market_type": "BSE",
                }
            )

            if created:
                added.append((row_id, symbol))
                print(f"✅ Added: Row {row_id} — {symbol}")
            else:
                skipped.append((row_id, symbol))
                print(f"ℹ️ Already exists: {symbol}")

    print("\n==================== IMPORT SUMMARY ====================")
    print(f"✔ Added symbols ({len(added)}): {added}")
    print(f"❌ Skipped symbols ({len(skipped)}): {skipped}")
    print("========================================================\n")

    return added, skipped


def import_bse_symbols_from_csv(csv_name="Analyst_Workday.csv"):
    csv_path = os.path.join(
        settings.BASE_DIR,
        "marketdata",
        "utils",
        "data",
        csv_name
    )

    print(f"\n📄 Using CSV file: {csv_path}\n")

    added_bse = []
    skipped = []
    not_in_bse = []

    # Ensure logs folder exists
    logs_dir = os.path.join(
        settings.BASE_DIR,
        "marketdata",
        "utils",
        "data",
        "logs"
    )
    os.makedirs(logs_dir, exist_ok=True)

    # Timestamp for filenames
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Read CSV file
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            row_id = row[CSV_COLUMNS["id"]].strip()
            symbol = row[CSV_COLUMNS["symbol"]].strip()
            sector_name = row[CSV_COLUMNS["sector"]].strip()
            company = row[CSV_COLUMNS["company_name"]].strip()

            # Skip missing fields
            if not symbol or not sector_name or not company:
                skipped.append((row_id, symbol))
                print(f"⛔ Skipped row {row_id}: Missing required fields")
                continue

            # Convert NSE symbol to BSE version
            bse_symbol = f"{symbol}_BSE"

            # Check if BSE symbol is available
            if bse_symbol not in AVAILABLE_BSE:
                not_in_bse.append((row_id, bse_symbol))
                print(f"⚠️ Not in BSE list: {bse_symbol}")
                continue

            # Ensure sector exists
            sector_obj, _ = Sectors.objects.get_or_create(name=sector_name)

            # Insert into DB
            obj, created = Symbol.objects.get_or_create(
                symbol=bse_symbol,
                defaults={
                    "company_name": company,
                    "sector": sector_obj,
                    "market_type": "BSE",
                }
            )

            if created:
                added_bse.append((row_id, bse_symbol))
                print(f"✅ Added BSE: {bse_symbol}")
            else:
                print(f"ℹ️ Already exists: {bse_symbol}")

    # -------------------------
    # Write 3 SEPARATE LOG FILES
    # -------------------------

    added_file = os.path.join(logs_dir, f"added_bse_{timestamp}.txt")
    skipped_file = os.path.join(logs_dir, f"skipped_bse_{timestamp}.txt")
    not_in_bse_file = os.path.join(logs_dir, f"not_available_bse_{timestamp}.txt")

    # Write Added
    with open(added_file, "w", encoding="utf-8") as f:
        f.write("✔ Added BSE Symbols:\n")
        for entry in added_bse:
            f.write(f"{entry}\n")

    # Write Skipped
    with open(skipped_file, "w", encoding="utf-8") as f:
        f.write("❌ Skipped (Missing Fields):\n")
        for entry in skipped:
            f.write(f"{entry}\n")

    # Write Not Available in BSE list
    with open(not_in_bse_file, "w", encoding="utf-8") as f:
        f.write("⚠️ Not Available in Official BSE Symbol List:\n")
        for entry in not_in_bse:
            f.write(f"{entry}\n")

    # Summary
    print("\n==================== IMPORT SUMMARY ====================")
    print(f"✔ Added BSE ({len(added_bse)}): {added_bse[:5]} ...")
    print(f"❌ Skipped ({len(skipped)}): {skipped[:5]} ...")
    print(f"⚠️ Not in BSE list ({len(not_in_bse)}): {not_in_bse[:5]} ...")
    print("\n📝 Log files saved:")
    print(f"   → {added_file}")
    print(f"   → {skipped_file}")
    print(f"   → {not_in_bse_file}")
    print("========================================================\n")

    return added_bse, skipped, not_in_bse