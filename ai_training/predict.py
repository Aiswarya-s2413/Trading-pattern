
import torch
import pandas as pd
import numpy as np
import os
import sys
import django
from datetime import timedelta

# specific to potential mac issues with matplotlib/plotting backends if imported later
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Setup Django Environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'my_project.settings')
django.setup()

from marketdata.models import Symbol, Parameter, EodPrice, Sectors
from model import HybridLSTMAttention

# Parameters (Must match training)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
METADATA_SIZE = 1
INPUT_SIZE = 1
SEQUENCE_LENGTH = 30

def get_single_sector_confidence(sector_name):
    """
    Calculates confidence for a single sector on the fly.
    """
    try:
        sector = Sectors.objects.get(name=sector_name)
    except Sectors.DoesNotExist:
        return 0.5
        
    symbols = Symbol.objects.filter(sector=sector)[:5] 
    if not symbols:
        return 0.5
        
    success_count = 0
    total_count = 0
    
    for symbol in symbols:
        prices = list(EodPrice.objects.filter(symbol=symbol).order_by('-trade_date')[:60])
        if len(prices) < 20:
            continue
            
        for i in range(len(prices) - 10):
            current_price = float(prices[i+10].close)
            future_price = float(prices[i].close)
            
            if current_price > 0:
                 if future_price >= current_price * 1.05:
                     success_count += 1
                 total_count += 1
                 
    if total_count == 0:
        return 0.5
        
    return success_count / total_count

def predict(symbol_name, model_path='best_model.pth'):
    # 1. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridLSTMAttention(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        metadata_size=METADATA_SIZE
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please train the model first.")
        return
        
    model.eval()
    
    # 2. Fetch Data
    try:
        symbol = Symbol.objects.get(symbol=symbol_name)
    except Symbol.DoesNotExist:
        print(f"Symbol {symbol_name} not found.")
        return

    print(f"Predicting for {symbol.symbol} (Sector: {symbol.sector.name if symbol.sector else 'Unknown'})...")
    
    # Get last SEQUENCE_LENGTH RSC values
    params = Parameter.objects.filter(symbol=symbol).order_by('-trade_date')[:SEQUENCE_LENGTH]
    
    if len(params) < SEQUENCE_LENGTH:
        print(f"Not enough data for prediction. Need {SEQUENCE_LENGTH} days, got {len(params)}.")
        return
        
    # Order is descending (newest first), so we need to reverse it to be chronological
    params = list(params)[::-1]
    
    rsc_values = [float(p.rsc30) if p.rsc30 is not None else 0.0 for p in params]
    
    # 3. Preprocess
    # Fill None/NaNs if any (simple forward fill logic or 0)
    # In this case we just used 0.0 default above.
    
    sequence = torch.tensor(rsc_values, dtype=torch.float32).view(1, SEQUENCE_LENGTH, 1).to(device)
    
    # Sector Confidence
    sector_conf = 0.5
    if symbol.sector:
        sector_conf = get_single_sector_confidence(symbol.sector.name)
        
    meta = torch.tensor([sector_conf], dtype=torch.float32).view(1, 1).to(device)
    
    # 4. Inference
    with torch.no_grad():
        output = model(sequence, meta)
        probability = torch.sigmoid(output).item()
        
    print(f"Prediction result for {symbol_name}:")
    print(f"  Sector Confidence: {sector_conf:.2f}")
    print(f"  Success Probability: {probability:.4f} ({probability*100:.2f}%)")
    
    return probability

if __name__ == "__main__":
    if len(sys.argv) > 1:
        s_name = sys.argv[1]
        predict(s_name)
    else:
        # Test with a dummy symbol or pick one from DB
        first_symbol = Symbol.objects.first()
        if first_symbol:
            predict(first_symbol.symbol)
        else:
            print("No symbols found in database.")
