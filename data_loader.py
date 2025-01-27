import yfinance as yf
import pandas as pd

class DataLoader:
    def __init__(self):
        pass  # No API key needed for yfinance
        
    def load_data(self, symbols, start_date, end_date):
        """Load historical data from Yahoo Finance with validation"""
        data = yf.download(
            symbols,
            start=start_date,
            end=end_date,
            group_by='ticker',
            auto_adjust=True  # Use adjusted prices
        )
        
        # Handle multi-index columns
        if len(symbols) > 1:
            merged = pd.concat(
                [data[symbol].add_prefix(f"{symbol}_") for symbol in symbols],
                axis=1
            )
        else:
            merged = data.add_prefix(f"{symbols[0]}_")
            
        # Data validation and cleaning
        merged = self._validate_prices(merged)
        merged = self._clean_volumes(merged)
        merged = self._calculate_indicators(merged)
        return merged.dropna()

    def _validate_prices(self, data):
        """Fix Yahoo Finance data anomalies"""
        # Replace zero/negative prices with NaN
        price_cols = [col for col in data.columns if 'Close' in col]
        data[price_cols] = data[price_cols].mask(data[price_cols] <= 0)
        
        # Forward fill missing prices
        return data.ffill().dropna()

    def _clean_volumes(self, data):
        """Handle unrealistic volume values"""
        vol_cols = [col for col in data.columns if 'Volume' in col]
        
        # Cap volume at 3 standard deviations from mean
        for col in vol_cols:
            mean = data[col].mean()
            std = data[col].std()
            cap = mean + 3*std
            data[col] = data[col].clip(upper=cap)
            
        return data

    def _calculate_indicators(self, data):
        """Updated for Yahoo Finance column names"""
        # Calculate for each symbol
        for col in data.columns:
            if 'Close' in col:
                symbol = col.split('_')[0]
                
                # RSI
                close_series = data[col]
                delta = close_series.diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(14).mean()
                avg_loss = loss.rolling(14).mean()
                rs = avg_gain / avg_loss
                data[f'{symbol}_RSI'] = 100 - (100 / (1 + rs))
                
                # MACD
                exp1 = close_series.ewm(span=12, adjust=False).mean()
                exp2 = close_series.ewm(span=26, adjust=False).mean()
                data[f'{symbol}_MACD'] = exp1 - exp2
                data[f'{symbol}_Signal'] = data[f'{symbol}_MACD'].ewm(span=9, adjust=False).mean()
        
        # Normalize
        return (data - data.mean()) / data.std()
