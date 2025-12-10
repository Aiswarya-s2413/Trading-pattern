import csv
import os
from django.conf import settings
from marketdata.models import Symbol

def generate_missing_report():
    print("Starting missing EOD report generation...")
    # Filter symbols that do not have any eodprice associated
    missing_data_symbols = Symbol.objects.filter(eodprice__isnull=True)
    
    output_dir = os.path.join(settings.BASE_DIR, 'marketdata', 'utils', 'data', 'report')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'missing_eod_stocks.csv')
    
    count = 0
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Symbol', 'Sector', 'Market Type'])
        for sym in missing_data_symbols:
            sector_name = sym.sector.name if sym.sector else 'N/A'
            writer.writerow([sym.symbol, sector_name, sym.market_type])
            count += 1
            
    print(f"Successfully generated missing data report for {count} symbols.")
    print(f"File saved to: {output_file}")

def generate_present_report():
    print("Starting present EOD report generation...")
    # Filter symbols that HAVE eodprice associated. Distinct is needed as join can return duplicates.
    present_data_symbols = Symbol.objects.filter(eodprice__isnull=False).distinct()
    
    output_dir = os.path.join(settings.BASE_DIR, 'marketdata', 'utils', 'data', 'report')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'present_eod_stocks.csv')
    
    count = 0
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Symbol', 'Sector', 'Market Type'])
        
        # Use iterator to handle large datasets efficiently
        for sym in present_data_symbols.iterator():
            sector_name = sym.sector.name if sym.sector else 'N/A'
            writer.writerow([sym.symbol, sector_name, sym.market_type])
            count += 1
            
    print(f"Successfully generated present data report for {count} symbols.")
    print(f"File saved to: {output_file}")

def generate_comprehensive_report():
    print("Starting comprehensive EOD report generation...")
    from django.db.models import Min, Max, Count
    
    # Annotate symbols with necessary aggregation data
    # We use 'eodprice' which is the default related name for ForeignKey if not specified related_name
    symbols_data = Symbol.objects.annotate(
        start_date=Min('eodprice__trade_date'),
        end_date=Max('eodprice__trade_date'),
        total_count=Count('eodprice')
    )
    
    output_dir = os.path.join(settings.BASE_DIR, 'marketdata', 'utils', 'data', 'report')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    output_file = os.path.join(output_dir, 'comprehensive_eod_report.csv')
    
    count = 0
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Symbol', 'Sector', 'Market Type', 'Is Available', 'Start Date', 'End Date', 'Total Count', 'Range (Years)'])
        
        # Use iterator to avoid loading everything into memory if list is huge
        # But aggregate queries sometimes don't work well with iterator() depending on DB, 
        # for SQLite/Postgres it's usually fine. Safest is just loop if not massive (3k is small).
        for sym in symbols_data:
            sector_name = sym.sector.name if sym.sector else 'N/A'
            is_available = sym.total_count > 0
            
            start_date = sym.start_date if sym.start_date else ''
            end_date = sym.end_date if sym.end_date else ''
            total_count = sym.total_count
            
            range_years = 0.0
            if is_available and sym.start_date and sym.end_date:
                delta = sym.end_date - sym.start_date
                range_years = round(delta.days / 365.25, 2)
            
            writer.writerow([
                sym.symbol, 
                sector_name, 
                sym.market_type, 
                is_available, 
                start_date, 
                end_date, 
                total_count, 
                range_years
            ])
            count += 1
            
    print(f"Successfully generated comprehensive report for {count} symbols.")
    print(f"File saved to: {output_file}")

if __name__ == "__main__":
    # generate_missing_report()
    # generate_present_report()
    generate_comprehensive_report()
