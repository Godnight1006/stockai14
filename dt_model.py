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
    def __init__(self, observation_space, act_dim, hidden_size=128):
        super().__init__()
        state_dim = observation_space.shape[0]
        self.state_embed = nn.Linear(state_dim, hidden_size)
        self.action_embed = nn.Embedding(act_dim, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size*4
            ),
            num_layers=3
        )
        
        self.predict_action = nn.Linear(hidden_size, act_dim)
        self.features_dim = hidden_size  # Required by SB3 for features extractors
        
    def forward(self, states, actions, timesteps):
        # Embed inputs
        state_emb = self.state_embed(states)
        action_emb = self.action_embed(actions)
        
        # Combine embeddings
        seq_len = states.size(0)
        combined = state_emb + action_emb
        
        # Add positional encoding
        combined = self.positional_encoding(combined)
        
        # Transformer processing
        transformer_out = self.transformer(combined)
        
        # Predict next action
        action_logits = self.predict_action(transformer_out[-1])
        return action_logits
