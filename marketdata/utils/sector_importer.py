import csv
import os
from django.conf import settings
from marketdata.models import Sectors

BASE_DIR = settings.BASE_DIR

def import_sectors_from_csv():
    """
    Reads a CSV file, extracts unique sectors, and saves them in Sector model.
    Reusable across shell, commands, scripts.
    """
    
    csv_path = os.path.join(
        BASE_DIR,
        "marketdata",
        "utils",
        "data",
        "Analyst_Workday.csv"
    )
    unique_sectors = set()

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            sector_name = row[2].strip()  # 3rd column
            if sector_name:
                unique_sectors.add(sector_name)

    print(f"Found {len(unique_sectors)} unique sectors.")

    for sector in unique_sectors:
        obj, created = Sectors.objects.get_or_create(name=sector)
        if created:
            print(f"Added sector: {sector}")
        else:
            print(f"Already exists: {sector}")

    print("Import complete!")
    return unique_sectors