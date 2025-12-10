
import os

TRUEDATA_FILE = r"d:\Work\Notion Trading\32.ALL_BSE_EQ.txt"
FAILED_FILE = r"d:\Work\Notion Trading\BSE_Failed.txt"
OUTPUT_DIR = r"d:\Work\Notion Trading\prodigy-backend\marketdata\utils\data\logs"

def analyze():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Reading TrueData symbols from {TRUEDATA_FILE}...")
    try:
        with open(TRUEDATA_FILE, 'r') as f:
            # Strip whitespace and ignore empty lines
            truedata_symbols = set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        print(f"Error: TrueData file not found at {TRUEDATA_FILE}")
        return

    print(f"Total TrueData Symbols: {len(truedata_symbols)}")

    print(f"Reading Failed symbols from {FAILED_FILE}...")
    try:
        with open(FAILED_FILE, 'r') as f:
            failed_symbols = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Failed file not found at {FAILED_FILE}")
        return

    print(f"Total Failed Symbols: {len(failed_symbols)}")

    listed_but_failed = []
    not_listed = []

    for sym in failed_symbols:
        if sym in truedata_symbols:
            listed_but_failed.append(sym)
        else:
            not_listed.append(sym)

    print("-" * 30)
    print(f"Failed Symbols FOUND in TrueData List: {len(listed_but_failed)}")
    print(f"Failed Symbols NOT FOUND in TrueData List: {len(not_listed)}")
    print("-" * 30)

    # Write results
    listed_path = os.path.join(OUTPUT_DIR, "bse_failed_but_listed.txt")
    not_listed_path = os.path.join(OUTPUT_DIR, "bse_failed_not_listed.txt")
    reasons_path = os.path.join(OUTPUT_DIR, "bse_listed_failure_reasons.txt")
    failed_log_path = os.path.join(OUTPUT_DIR, "failed_symbols.txt")

    # Load error messages
    symbol_errors = {}
    if os.path.exists(failed_log_path):
        print(f"Reading error logs from {failed_log_path}...")
        with open(failed_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split(" — ", 1)
                if len(parts) == 2:
                    symbol_errors[parts[0].strip()] = parts[1].strip()
    else:
        print(f"Warning: Failed log file not found at {failed_log_path}")

    with open(listed_path, 'w') as f:
        for sym in listed_but_failed:
            f.write(sym + "\n")
    
    with open(reasons_path, 'w') as f:
        f.write(f"Analysis of {len(listed_but_failed)} symbols that are in TrueData list but failed:\n")
        f.write("-" * 50 + "\n")
        for sym in listed_but_failed:
            err = symbol_errors.get(sym, "Reason not found in logs")
            f.write(f"{sym}: {err}\n")

    with open(not_listed_path, 'w') as f:
        for sym in not_listed:
            f.write(sym + "\n")

    print(f"Saved 'Listed but Failed' symbols to: {listed_path}")
    print(f"Saved 'Listed Failure Reasons' to: {reasons_path}")
    print(f"Saved 'Not Listed' symbols to: {not_listed_path}")

if __name__ == "__main__":
    analyze()
