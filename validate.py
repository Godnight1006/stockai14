from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from data_loader import DataLoader
from trading_env import StockTradingEnv

def fetch_2024_data():
    loader = DataLoader(api_key='YOUR_API_KEY')
    return loader.load_data(['AAPL', 'MSFT', 'GOOG'], '2024-01-01', '2024-12-31')

def run_validation():
    # Load trained model
    model = MaskablePPO.load("dt_stock_trader")
    
    # Load validation data
    val_df = fetch_2024_data()
    
    # Create validation environment
    env = StockTradingEnv(val_df, initial_balance=100000)
    
    # Run validation
    obs = env.reset()
    while True:
        action_masks = get_action_masks(env)
        action, _ = model.predict(obs, action_masks=action_masks)
        obs, _, done, _ = env.step(action)
        
        if done:
            break
            
    # Calculate final portfolio value
    current_price = val_df.iloc[-1]['4. close']
    final_value = env.balance + env.shares_held * current_price
    print(f"Final portfolio value: ${final_value:,.2f}")

if __name__ == "__main__":
    run_validation()
