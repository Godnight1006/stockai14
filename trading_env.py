import gymnasium as gym
import numpy as np

class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, window_size=1260):  # ~5 years of trading days
        super().__init__()
        self.df = df
        self.initial_balance = initial_balance
        
        # Extract symbols from column names
        self.symbols = list({col.split('_')[0] for col in df.columns})
        self.num_stocks = len(self.symbols)
        
        # New action space: 3 actions per stock (hold/buy/sell)
        self.action_space = gym.spaces.Discrete(3 * self.num_stocks)
        
        # New observation space with historical window
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(window_size, len(df.columns)),  # 2D observation
            dtype=np.float32
        )
        self.window_size = window_size
        
        # Track positions per stock
        self.shares_held = {symbol: 0 for symbol in self.symbols}
        self.reset()
        
    def reset(self, seed=None, options=None):
        self.balance = self.initial_balance
        self.shares_held = {symbol: 0 for symbol in self.symbols}
        # Start at window_size to have enough history
        self.current_step = self.window_size  
        return self._get_obs(), {}
    
    def step(self, action):
        # Decompose action into stock_idx and action_type
        stock_idx = action // 3
        action_type = action % 3  # 0=hold, 1=buy, 2=sell
        symbol = self.symbols[stock_idx]
        
        # Get current price for selected stock
        current_price = self.df.iloc[self.current_step][f"{symbol}_Close"]
        
        # Execute trade for specific stock
        if action_type == 1:
            self._execute_buy(current_price, symbol)
        elif action_type == 2:
            self._execute_sell(current_price, symbol)
            
        # Calculate portfolio value across all stocks
        portfolio_value = self.balance
        for sym in self.shares_held:
            price = self.df.iloc[self.current_step][f"{sym}_Close"]
            portfolio_value += self.shares_held[sym] * price
            
        reward = portfolio_value - self.initial_balance
        
        # Update step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        return self._get_obs(), reward, done, False, {}
    
    def _get_obs(self):
        window = self.df.iloc[self.current_step-self.window_size:self.current_step]
        # Add epsilon to prevent division by zero
        return ((window - window.mean()) / (window.std() + 1e-8)).values
    
    def _execute_buy(self, price, symbol):
        if self.balance >= price:
            self.shares_held[symbol] += 1
            self.balance -= price
            
    def _execute_sell(self, price, symbol):
        if self.shares_held[symbol] > 0:
            self.shares_held[symbol] -= 1
            self.balance += price
            
    def action_masks(self):
        masks = []
        for action in range(self.action_space.n):
            stock_idx = action // 3
            action_type = action % 3
            symbol = self.symbols[stock_idx]
            masks.append(
                action_type != 2 or self.shares_held[symbol] > 0
            )
        return masks
