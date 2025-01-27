import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries

class DataLoader:
    def __init__(self, api_key):
        self.ts = TimeSeries(key=api_key, output_format='pandas')
        
    def load_data(self, symbols, start_date, end_date):
        """Load historical data for multiple symbols and calculate indicators"""
        merged_data = pd.DataFrame()
        for symbol in symbols:
            data, _ = self.ts.get_daily_adjusted(symbol, outputsize='full')
            data = data.loc[start_date:end_date]
            data = self._calculate_indicators(data)
            # Add symbol prefix to columns
            data = data.add_prefix(f"{symbol}_")
            merged_data = pd.concat([merged_data, data], axis=1)
        return merged_data.ffill().dropna()
    
    def _calculate_indicators(self, data):
        """Calculate technical indicators"""
        # RSI
        delta = data['4. close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['4. close'].ewm(span=12, adjust=False).mean()
        exp2 = data['4. close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Normalize all values
        data = (data - data.mean()) / data.std()
        return data
