
import os
import sys
import django
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import timedelta

# specific to potential mac issues with matplotlib/plotting backends if imported later
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Setup Django Environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'my_project.settings')
django.setup()

from marketdata.models import Symbol, Parameter, EodPrice, Sectors

class StockDataset(Dataset):
    def __init__(self, sequences, sector_confidences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.sector_confidences = torch.tensor(sector_confidences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.sector_confidences[idx], self.labels[idx]

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_sector_confidence():
    """
    Calculates the historical success rate for each sector.
    Success = >5% gain in 10 days.
    Returns: Dict {sector_name: confidence_score}
    """
    print("Calculating Sector Confidence...")
    sectors = Sectors.objects.all()
    sector_confidence = {}
    
    # Heuristic: Processing EVERY EOD price for EVERY symbol is too slow.
    # We will sample or use a simplified metric for now, or just placeholders if data is massive.
    # Ideally, this should be pre-calculated and stored.
    # For this implementation, let's look at the last 6 months of data for active symbols.
    
    # TODO: Optimize this for production with a SQL query or pre-aggregation
    # Current implementation: Placeholder / Simplified logic
    for sector in sectors:
        # Get random sample of symbols in this sector to estimate confidence
        symbols = Symbol.objects.filter(sector=sector)[:5] # Limit to 5 symbols per sector for speed
        if not symbols:
            sector_confidence[sector.name] = 0.5
            continue
            
        success_count = 0
        total_count = 0
        
        for symbol in symbols:
            prices = list(EodPrice.objects.filter(symbol=symbol).order_by('-trade_date')[:100]) # Last 100 days
            if len(prices) < 20:
                continue
            
            # Simple check: how many times did price go up 5% in next 10 days?
            # Iterating backwards
            for i in range(len(prices) - 10):
                current_price = float(prices[i+10].close) # i is future, i+10 is past (reverse order)
                future_price = float(prices[i].close)     # Wait, order_by('-trade_date') means index 0 is NEWEST.
                                                    # So prices[i] is newer than prices[i+10].
                                                    # We want: if we buy at prices[i+10], is prices[i] > 1.05 * prices[i+10]?
                
                if current_price > 0:
                     if future_price >= current_price * 1.05:
                         success_count += 1
                     total_count += 1
        
        if total_count > 0:
            # We store BOTH (Success Rate, Total Count)
            sector_confidence[sector.name] = (success_count / total_count, total_count)
        else:
            sector_confidence[sector.name] = (0.5, 0) # Neutral
            
    print(f"Sector Confidence Calculated: {sector_confidence}")
    return sector_confidence

def fetch_and_process_data(sequence_length=30, prediction_window=10, target_symbols=None):
    """
    Fetches data from DB, processes it into sequences for LSTM.
    Returns:
        sequences: np.array of shape (num_samples, sequence_length, features)
        meta_features: np.array of shape (num_samples, meta_dim)
        labels: np.array of shape (num_samples,)
        dates: np.array of shape (num_samples,) - date of prediction (t)
        symbols_list: np.array of shape (num_samples,)
        pos_weight: float - class weight for imbalance
    """
    
    # 1. Fetch Data
    # We need RSC (Parameter) and Price (EodPrice)
    # We also need Sector Confidence (Calculated from historicals)
    
    sector_confidence_map = get_sector_confidence()
    
    sequences = []
    meta_features = []
    labels = []
    dates = []
    symbols_list = []
    reliability_scores = []
    
    if target_symbols:
        symbols = Symbol.objects.filter(symbol__in=target_symbols)
        print(f"Fetching data for specific symbols: {target_symbols}")
    else:
        # Fetch a subset of symbols for training to keep it manageable
        # For backtesting, we might want ALL symbols, or a larger subset. 
        # Let's keep it restricted for now to avoid OOM, but maybe increase to 50?
        symbols = Symbol.objects.filter(sector__isnull=False)[:50] 
    
    for symbol in symbols:
        print(f"Processing {symbol.symbol}...")
        # Fetch Parameters (RSC) and Prices
        params = Parameter.objects.filter(symbol=symbol).order_by('trade_date')
        prices = EodPrice.objects.filter(symbol=symbol).order_by('trade_date')
        
        if not params.exists() or not prices.exists():
            continue
            
        # Convert to DataFrames
        df_params = pd.DataFrame(list(params.values('trade_date', 'rsc30')))
        df_prices = pd.DataFrame(list(prices.values('trade_date', 'close')))
        
        # Convert Decimals to float for pandas operations
        if not df_params.empty:
            df_params['rsc30'] = df_params['rsc30'].astype(float)
        if not df_prices.empty:
            df_prices['close'] = df_prices['close'].astype(float)
        
        # Merge on trade_date
        df = pd.merge(df_params, df_prices, on='trade_date')
        df.sort_values('trade_date', inplace=True)
        
        if len(df) < sequence_length + prediction_window:
            continue
        
        # Calculate Labels
        # Label = 1 if Close[t+10] > Close[t] * 1.05
        # We want the date at t+prediction_window (the prediction date) or t (the trade date)?
        # Usually for backtest report: "On Date T, we predicted Outcome at T+WINDOW".
        # Let's record Date T (the date we make the prediction).
        
        df['target'] = (df['close'].shift(-prediction_window) > df['close'] * 1.05).astype(int)
        
        # Fill NaNs in RSC (if any)
        df['rsc30'] = df['rsc30'].ffill().fillna(0)
        
        # Normalization
        # Use simple MinMax Scaling for RSC within the dataframe for now, or global if we knew global min/max
        # Global is better for consistency. Let's assume RSC is ratio around 1.0 (e.g. 0.5 to 2.0). 
        # But for 'rsc30' it might be different. Let's inspect max value or just use standard scaling.
        # Let's use standard scaling (z-score) per symbol for now (local normalization) to capture relative momentum?
        # Or Global? Global is safer for comparing stocks.
        # Let's use a robust scaler approach: (x - median) / IQR, or just (x - mean) / std.
        
        if df['rsc30'].std() != 0:
            df['rsc30'] = (df['rsc30'] - df['rsc30'].mean()) / df['rsc30'].std()
        else:
            df['rsc30'] = 0
            
        data = df['rsc30'].values
        targets = df['target'].values
        trade_dates = df['trade_date'].values
        
        sector_name = symbol.sector.name
        # Updated get_sector_confidence returns (Success Rate, Count)
        conf_tuple = sector_confidence_map.get(sector_name, (0.5, 0))
        
        # Ensure it is a tuple, in case legacy code or fallback returns float
        if isinstance(conf_tuple, tuple):
             sec_conf = conf_tuple[0]
             trade_count = conf_tuple[1]
        else:
             sec_conf = conf_tuple # Should be float
             trade_count = 0
             
        # Calculate user's Logarithmic Reliability Score: ln(count+1)/ln(30+1)
        # We clamp it to 0-1 range (though it might exceed if count > 30, user mentioned >30 is 1.0 max)
        # Ah user said "30+ trades: Score = 1.0 (Max)"
        reliability_score = np.log(trade_count + 1) / np.log(31) if trade_count < 30 else 1.0
        
        # Create Sequences
        for i in range(len(data) - sequence_length - prediction_window):
            seq = data[i : i + sequence_length]
            label = targets[i + sequence_length] # The label associated with the future outcome
            date_of_prediction = trade_dates[i + sequence_length] # The date we have the full sequence and make prediction
            
            # Reshape seq for LSTM [seq_len, features] -> here features=1
            sequences.append(seq.reshape(-1, 1))
            
            # Metadata: We store [Sector Confidence, Reliability Score]
            # But wait, original model only expects 1 metadata field (sec_conf).
            # If we add reliability_score to meta_features, `metas` will have shape (N, 2).
            # If `model.py` `METADATA_SIZE` is 1, this breaks `forward` pass (concatenation).
            # So let's keep `metas` as just sec_conf for the model, and store `reliability_score` separately 
            # for the Excel sheet, unless we retrain model with size=2.
            # User just wants "new excel sheet". Let's avoid breaking model.
            meta_features.append(sec_conf)
            labels.append(label)
            dates.append(date_of_prediction)
            symbols_list.append(symbol.symbol)
            
            # Wait, how do pass reliability_score? 
            # Let's create a separate list `reliability_scores`.
            reliability_scores.append(reliability_score)
            
    # Calculate pos_weight for imbalance
    # pos_weight = (num_negatives / num_positives)
    labels_np = np.array(labels)
    num_pos = np.sum(labels_np == 1)
    num_neg = np.sum(labels_np == 0)
    pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    print(f"Computed class weight (pos_weight): {pos_weight:.4f}")
            
    return np.array(sequences), np.array(meta_features), labels_np, np.array(dates), np.array(symbols_list), pos_weight, np.array(reliability_scores)

def get_dataloaders(batch_size=32):
    # Retrieve all data (unpack 7 values)
    # We ignore reliability_scores for training
    sequences, metas, labels, dates, symbols, pos_weight, _ = fetch_and_process_data()
    
    # Split Train/Val (Simple time-series split on the whole dataset is tricky if mixed symbols)
    # But for general training, random split or simple valid split is okay.
    # We will just ignore dates/symbols for the generic train loop.
    
    train_size = int(0.8 * len(sequences))
    
    train_seq = sequences[:train_size]
    train_meta = metas[:train_size]
    train_labels = labels[:train_size]
    
    val_seq = sequences[train_size:]
    val_meta = metas[train_size:]
    val_labels = labels[train_size:]
    
    train_dataset = StockDataset(train_seq, train_meta, train_labels)
    val_dataset = StockDataset(val_seq, val_meta, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, pos_weight

if __name__ == "__main__":
    train_loader, val_loader, pos_weight = get_dataloaders()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
