import gymnasium as gym
import numpy as np

class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super().__init__()
        self.df = df
        self.initial_balance = initial_balance
        
        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = gym.spaces.Discrete(3)
        
        # Observation space: OHLCV + indicators
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(len(df.columns),), dtype=np.float32)
        
        self.reset()
        
    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        return self._get_obs()
    
    def step(self, action):
        current_price = self.df.iloc[self.current_step]['4. close']
        
        # Execute trade
        if action == 1:  # Buy
            self._execute_buy(current_price)
        elif action == 2:  # Sell
            self._execute_sell(current_price)
            
        # Calculate reward
        portfolio_value = self.balance + self.shares_held * current_price
        reward = portfolio_value - self.initial_balance
        
        # Update step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        return self._get_obs(), reward, done, {}
    
    def _get_obs(self):
        return self.df.iloc[self.current_step].values
    
    def _execute_buy(self, price):
        if self.balance >= price:
            self.shares_held += 1
            self.balance -= price
            
    def _execute_sell(self, price):
        if self.shares_held > 0:
            self.shares_held -= 1
            self.balance += price
            
    def action_masks(self):
        # Prevent selling when no shares are held
        return [True, True, self.shares_held > 0]
