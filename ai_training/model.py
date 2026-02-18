
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_size]
        
        # Calculate attention weights
        # shape: [batch_size, seq_len, 1]
        attn_weights = F.softmax(self.attention(lstm_output), dim=1)
        
        # Calculate context vector using attention weights
        # shape: [batch_size, hidden_size]
        context_vector = torch.sum(attn_weights * lstm_output, dim=1)
        
        return context_vector, attn_weights

class HybridLSTMAttention(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, metadata_size=1,dropout=0.2):
        super(HybridLSTMAttention, self).__init__()
        
        # Stream A: Sequential Data (RSC)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention Mechanism
        self.attention = Attention(hidden_size)
        
        # Stream B: Metadata (Sector Confidence)
        # We concatenate the context vector (from Attention) with the metadata
        self.fc1 = nn.Linear(hidden_size + metadata_size, 32)
        self.fc2 = nn.Linear(32, 1) # Output: Logits (use Sigmoid in training/inference if needed, or BCEWithLogitsLoss)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, metadata):
        """
        x: [batch_size, seq_len, 1] (RSC values)
        metadata: [batch_size] or [batch_size, 1] (Sector Confidence)
        """
        
        # Ensure metadata has correct shape [batch_size, 1]
        if metadata.dim() == 1:
            metadata = metadata.unsqueeze(1)
            
        # 1. Process Sequence Data through LSTM
        lstm_out, (hn, cn) = self.lstm(x) # lstm_out: [batch, seq_len, hidden]
        
        # 2. Apply Attention
        context_vector, attn_weights = self.attention(lstm_out) # context: [batch, hidden]
        
        # 3. Concatenate with Metadata
        combined = torch.cat((context_vector, metadata), dim=1) # [batch, hidden + meta]
        
        # 4. Fully Connected Layers
        out = F.relu(self.fc1(combined))
        out = self.dropout_layer(out)
        out = self.fc2(out) # [batch, 1]
        
        return out
