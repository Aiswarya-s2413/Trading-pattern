
import os
import sys
import django
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Setup Django Environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'my_project.settings')
django.setup()

from ai_training.data_loader import fetch_and_process_data, StockDataset
from ai_training.model import HybridLSTMAttention

# Parameters
BATCH_SIZE = 32
EPOCHS = 5 # Maybe less for backtest check, or standard 10
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
METADATA_SIZE = 1
INPUT_SIZE = 1

def run_backtest(target_symbols=None):
    print("Fetching all data for backtest...")
    sequences, metas, labels, dates, symbols, pos_weight_dummy, reliability_scores = fetch_and_process_data(target_symbols=target_symbols)
    
    # Convert dates to pandas datetime for easy filtering
    dates_pd = pd.to_datetime(dates)
    
    # Create masks
    # Train: Up to end of 2024
    train_mask = dates_pd.year <= 2024
    # Test: Year 2025
    test_mask = dates_pd.year == 2025
    
    X_train = sequences[train_mask]
    m_train = metas[train_mask]
    y_train = labels[train_mask]
    
    X_test = sequences[test_mask]
    m_test = metas[test_mask]
    y_test = labels[test_mask]
    dates_test = dates[test_mask]
    symbols_test = symbols[test_mask]
    # Filter reliability scores for test set
    reliability_scores_test = reliability_scores[test_mask]
    
    print(f"Train set size: {len(X_train)}")
    print(f"Test set size (2025): {len(X_test)}")
    
    if len(X_train) == 0:
        print("Error: No training data found <= 2024.")
        return
    if len(X_test) == 0:
        print("Error: No test data found for 2025.")
        return

    # Create DataLoaders
    train_dataset = StockDataset(X_train, m_train, y_train)
    test_dataset = StockDataset(X_test, m_test, y_test) # We use this for batch processing inference
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Calculate pos_weight for backtest training set
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    raw_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    
    # Clamp the weight to avoid exploding gradients or over-correction on small datasets
    # For single stock, high weight often forces model to predict 1 always.
    pos_weight = min(raw_pos_weight, 3.0) 
    
    print(f"Training set class weight sections: Raw={raw_pos_weight:.4f}, Used={pos_weight:.4f}")
    
    # Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = HybridLSTMAttention(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        metadata_size=METADATA_SIZE
    ).to(device)
    
    pos_weight_tensor = torch.tensor([pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training Loop
    print("Starting training on data <= 2024...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for seq, meta, label in train_loader:
            seq = seq.to(device)
            meta = meta.to(device)
            label = label.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(seq, meta)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")
        
    # Evaluation / Prediction on 2025
    print("Running predictions for 2025...")
    model.eval()
    
    results = []
    
    with torch.no_grad():
        # We iterate through test_loader OR just iterate index by index if we want easier mapping to symbols/dates
        # But using loader is faster for inference. We need to be careful to map back to meta info.
        # Since shuffle=False, the order matches X_test, dates_test, symbols_test.
        
        start_idx = 0
        for seq, meta, label in test_loader:
            seq = seq.to(device)
            meta = meta.to(device)
            
            outputs = model(seq, meta)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            actuals = label.cpu().numpy().flatten()
            
            batch_size_current = len(probs)
            
            # Map back to metadata
            batch_dates = dates_test[start_idx : start_idx + batch_size_current]
            batch_symbols = symbols_test[start_idx : start_idx + batch_size_current]
            # Meta contains sector confidence, let's extract it (it's the first col of meta)
            batch_conf = meta.cpu().numpy().flatten()
            
            # Extract batch reliability scores
            batch_reliability = reliability_scores_test[start_idx : start_idx + batch_size_current]
            
            for i in range(batch_size_current):
                results.append({
                    'Symbol': batch_symbols[i],
                    'Date': batch_dates[i],
                    'Sector_Confidence': batch_conf[i],
                    'Sample_Confidence': batch_reliability[i], # Added per user request
                    'Actual_Success': actuals[i],
                    'Predicted_Probability': probs[i],
                    # Use a higher threshold if we want to be more conservative? 
                    # Or just keep 0.5.                    'Predicted_Probability': probs[i],
                    # For full market, we can be more standard (0.5) or slightly conservative (0.6). 
                    # Let's use 0.5 as a baseline for the broad market model.
                    'Predicted_Label': 1 if probs[i] > 0.5 else 0
                })
            
            start_idx += batch_size_current

    # Save to Excel with two sheets
    df_results = pd.DataFrame(results)
    
    # Calculate Per-Stock Accuracy
    # Group by Symbol and calculate mean of (Predicted_Label == Actual_Success)
    stock_accuracy = df_results.groupby('Symbol').apply(
        lambda x: (x['Predicted_Label'] == x['Actual_Success']).mean()
    ).to_dict()
    
    # Map accuracy back to the main DataFrame
    df_results['Stock_Accuracy'] = df_results['Symbol'].map(stock_accuracy)
    
    output_file = 'backtest_results_2025_FULL.xlsx'
    
    # Create Explanation DataFrame
    explanation_data = {
        'Field Name': ['Symbol', 'Date', 'Sector_Confidence', 'Sample_Confidence', 'Predicted_Probability', 'Predicted_Label', 'Actual_Success', 'Stock_Accuracy'],
        'Description': [
            'Stock Ticker Symbol',
            'Date of Prediction',
            'Historical Success Rate of the Sector (Win Rate)',
            'Statistical Reliability Score (Based on Sample Size)',
            'AI Model Confidence Score (0-1)',
            'Final Decision (1=Buy, 0=Avoid)',
            'Actual Outcome (1=Success/Gain >5%, 0=Failure)',
            'The AI\'s "Track Record" for THIS specific stock in 2025'
        ],
        'Calculation': [
            '-',
            '-',
            'Winning Trades / Total Trades in Last 100 Days',
            'ln(Count + 1) / ln(31)',
            'Sigmoid(Neural Network Output)',
            'IF Probability > Threshold THEN 1 ELSE 0',
            'IF Price(T+10) > Price(T) * 1.05 THEN 1 ELSE 0',
            '(Correct Predictions for this Symbol) / (Total Days for this Symbol)'
        ]
    }
    df_explanation = pd.DataFrame(explanation_data)
    
    with pd.ExcelWriter(output_file) as writer:
        df_results.to_excel(writer, sheet_name='Backtest Results', index=False)
        df_explanation.to_excel(writer, sheet_name='Explanation', index=False)
        
    print(f"Backtest complete. Results saved to {output_file}")
    
    # Print summary metrics
    accuracy = (df_results['Actual_Success'] == df_results['Predicted_Label']).mean()
    print(f"2025 Overall Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    # Run full backtest (no target_symbols filter)
    run_backtest()
