
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
from data_loader import get_dataloaders
from model import HybridLSTMAttention

# Parameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
METADATA_SIZE = 1 # Sector Confidence
INPUT_SIZE = 1 # RSC Value

def train_model():
    # 1. Prepare Data
    print("Loading data...")
    train_loader, val_loader, pos_weight = get_dataloaders(batch_size=BATCH_SIZE)
    
    if len(train_loader) == 0:
        print("Not enough data to train. Exiting.")
        return

    # 2. Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = HybridLSTMAttention(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        metadata_size=METADATA_SIZE
    ).to(device)
    
    # Use pos_weight for class imbalance
    pos_weight_tensor = torch.tensor([pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    
    # 3. Training Loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for sequences, meta, labels in train_loader:
            sequences = sequences.to(device)
            meta = meta.to(device)
            labels = labels.to(device).unsqueeze(1) # [batch, 1]
            
            optimizer.zero_grad()
            
            outputs = model(sequences, meta)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 4. Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, meta, labels in val_loader:
                sequences = sequences.to(device)
                meta = meta.to(device)
                labels = labels.to(device).unsqueeze(1)
                
                outputs = model(sequences, meta)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.sigmoid(outputs).cpu().numpy()
                preds = (preds > 0.5).astype(int)
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate Metrics
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_precision = precision_score(all_labels, all_preds, zero_division=0)
        val_recall = recall_score(all_labels, all_preds, zero_division=0)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} "
              f"Val Loss: {avg_val_loss:.4f} "
              f"Acc: {val_accuracy:.4f} "
              f"Prec: {val_precision:.4f} "
              f"Recall: {val_recall:.4f}")
              
        # 5. Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("  -> Saved best model")

if __name__ == "__main__":
    train_model()
