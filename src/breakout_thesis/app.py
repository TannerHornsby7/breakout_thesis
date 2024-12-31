import yfinance as yf
import pandas as pd
from typing import Optional, Tuple, Dict
from datetime import datetime, date
import numpy as np

class MarketData:
    """Handles market data acquisition and preprocessing."""
    
    def __init__(self, data_source: str = 'yfinance'):
        """
        Initialize MarketData handler.
        
        Args:
            data_source (str): Source of market data ('yfinance', 'alpaca', etc.)
        """
        self.data_source = data_source
        
    def get_historical_data(
        self, 
        ticker: str, 
        start_date: date, 
        end_date: date
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical market data for a given ticker and date range.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (date): Start date for historical data
            end_date (date): End date for historical data
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with OHLCV data or None if fetch fails
        """
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                return None
            
            # Handle MultiIndex columns by selecting the first level only
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            return df
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None

class BreakoutAnalyzer:
    """Analyzes volume and price breakouts in market data."""
    
    def __init__(
        self,
        volume_ma_period: int = 20,
        volume_threshold: float = 2.0,
        price_threshold: float = 0.02,
        holding_period: int = 10
    ):
        """
        Initialize BreakoutAnalyzer with strategy parameters.
        
        Args:
            volume_ma_period (int): Period for volume moving average
            volume_threshold (float): Multiple of average volume to trigger breakout
            price_threshold (float): Minimum price change percentage
            holding_period (int): Number of days to hold after breakout
        """
        self.volume_ma_period = volume_ma_period
        self.volume_threshold = volume_threshold
        self.price_threshold = price_threshold
        self.holding_period = holding_period
        
    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate breakout signals and returns.
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: Original data with added signal and return columns
        """
        # Avoid lookahead bias by shifting volume MA
        df['vol_ma'] = df['Volume'].shift(1).rolling(self.volume_ma_period).mean()
        
        # Calculate daily returns without lookahead
        df['daily_return'] = df['Close'].pct_change()
        
        # Generate breakout signals
        df['volume_breakout'] = (
            df['Volume'] >= (self.volume_threshold * df['vol_ma'])
        )
        df['price_breakout'] = df['daily_return'] >= self.price_threshold
        df['is_breakout'] = df['volume_breakout'] & df['price_breakout']
        
        # Calculate forward returns for holding period
        df['entry_price'] = df['Close'].where(df['is_breakout'])
        df['exit_price'] = df['Close'].shift(-self.holding_period)
        df['strategy_return'] = (
            (df['exit_price'] - df['entry_price']) / df['entry_price']
        ).where(df['is_breakout'])
        
        return df
    
    def get_trade_summary(self, df: pd.DataFrame, ticker: str = 'AAPL') -> Dict:
        """
        Generate summary statistics for breakout trades.
        
        Args:
            df (pd.DataFrame): DataFrame with signals and returns
            ticker (str): Ticker symbol for labeling
        Returns:
            Dict: Summary statistics
        """
        trades = df[df['is_breakout']].copy()
        
        if len(trades) == 0:
            report = {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_return': 0.0,
                'sharpe_ratio': 0.0
            }
            if ticker:
                report['ticker'] = ticker
            return report
            
        win_rate = (trades['strategy_return'] > 0).mean()
        avg_return = trades['strategy_return'].mean()
        return_std = trades['strategy_return'].std()
        sharpe_ratio = (
            np.sqrt(252) * avg_return / return_std if return_std != 0 else 0
        )
        
        report = {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_return': avg_return,
            'sharpe_ratio': sharpe_ratio
        }
        
        if ticker:
            report['ticker'] = ticker
        return report

def main():
    """Main function for testing."""
    market_data = MarketData()
    analyzer = BreakoutAnalyzer()
    
    # Example usage
    ticker = "AAPL"
    start = date(2023, 1, 1)
    end = date(2024, 1, 1)
    
    df = market_data.get_historical_data(ticker, start, end)
    if df is not None:
        
        results = analyzer.calculate_signals(df)
        summary = analyzer.get_trade_summary(results, ticker)
        print(f"Strategy Results for {ticker}:")
        print(summary)

if __name__ == "__main__":
    main()