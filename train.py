from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from data_loader import DataLoader
from trading_env import StockTradingEnv
from dt_model import DecisionTransformer
import numpy as np

def load_preprocessed_data():
    loader = DataLoader()
    # Load multiple symbols
    return loader.load_data(['AAPL', 'MSFT', 'GOOG'], '2010-01-01', '2023-12-31')

def train_model():
    # Load and preprocess data
    df = load_preprocessed_data()
    
    # Create environment
    env = StockTradingEnv(df)
    
    # Configure model
    policy_kwargs = dict(
        features_extractor_class=DecisionTransformer,
        features_extractor_kwargs=dict(
            state_dim=env.observation_space.shape[0],
            act_dim=env.action_space.n
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
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2
    )
    
    # Train the model
    model.learn(total_timesteps=1_000_000)
    model.save("dt_stock_trader")

if __name__ == "__main__":
    train_model()
