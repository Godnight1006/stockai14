from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from data_loader import DataLoader
from trading_env import StockTradingEnv
from dt_model import DecisionTransformer
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_preprocessed_data():
    loader = DataLoader()
    # Load multiple symbols
    return loader.load_data(['AAPL', 'MSFT', 'GOOG'], '2010-01-01', '2023-12-31')

def train_model():
    # Load and preprocess data
    df = load_preprocessed_data()
    
    # Create environment with window size
    env = StockTradingEnv(df, window_size=1260)
    
    # Configure model
    policy_kwargs = dict(
        features_extractor_class=DecisionTransformer,
        features_extractor_kwargs=dict(
            act_dim=env.action_space.n,
            hidden_size=256  # Match new model size
        )
    )
    
    # Initialize and train model
    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=15,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2
    )
    
    # Configure for GPU training
    model.policy.to(device)
    torch.backends.cudnn.benchmark = True  # Enable cuDNN optimizations
    torch.set_float32_matmul_precision('high')  # For Tensor Cores
    
    print(f"Using GPUs: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Train the model
    model.learn(total_timesteps=1_000_000)
    model.save("dt_stock_trader")

if __name__ == "__main__":
    train_model()
