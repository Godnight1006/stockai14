import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class DecisionTransformer(nn.Module):
    def __init__(self, observation_space, act_dim, hidden_size=256):
        super().__init__()
        # Input is now (window_size, features)
        state_dim = observation_space.shape[1]  # features dimension
        
        # New embedding layer for sequence
        self.state_embed = nn.Linear(state_dim, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=16,  # increased attention heads
                dim_feedforward=hidden_size*4
            ),
            num_layers=6  # deeper network
        )
        self.features_dim = hidden_size  # Required by SB3 for features extractors
        
    def forward(self, states):
        # states shape: (batch_size, window_size, features)
        batch_size = states.shape[0]
        
        # Embed entire sequence
        state_emb = self.state_embed(states)  # (batch, window, hidden)
        state_emb = self.positional_encoding(state_emb)
        
        # Process through transformer
        transformer_out = self.transformer(state_emb)
        
        # Use last timestep's output for action prediction
        return transformer_out[:, -1, :]
