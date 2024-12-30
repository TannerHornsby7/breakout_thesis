import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from breakout_thesis.app import MarketData, BreakoutAnalyzer

@pytest.fixture
def sample_market_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
    
    np.random.seed(42)  # For reproducibility
    data = {
        'Open': np.random.randn(len(dates)) * 10 + 100,
        'High': np.random.randn(len(dates)) * 10 + 102,
        'Low': np.random.randn(len(dates)) * 10 + 98,
        'Close': np.random.randn(len(dates)) * 10 + 100,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }
    
    return pd.DataFrame(data, index=dates)

class TestMarketData:
    def test_initialization(self):
        md = MarketData()
        assert md.data_source == 'yfinance'
        
        md = MarketData('alpaca')
        assert md.data_source == 'alpaca'
    
    def test_get_historical_data(self):
        md = MarketData()
        start = date(2023, 1, 1)
        end = date(2024, 1, 1)
        
        # Test with valid ticker
        df = md.get_historical_data('AAPL', start, end)
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        
        # Test with invalid ticker
        df = md.get_historical_data('INVALID_TICKER', start, end)
        assert df is None

class TestBreakoutAnalyzer:
    def test_initialization(self):
        analyzer = BreakoutAnalyzer()
        assert analyzer.volume_ma_period == 20
        assert analyzer.volume_threshold == 2.0
        assert analyzer.price_threshold == 0.02
        assert analyzer.holding_period == 10
    
    def test_calculate_signals(self, sample_market_data):
        analyzer = BreakoutAnalyzer()
        results = analyzer.calculate_signals(sample_market_data)
        
        required_columns = [
            'vol_ma', 'daily_return', 'volume_breakout',
            'price_breakout', 'is_breakout', 'entry_price',
            'exit_price', 'strategy_return'
        ]
        
        # Check if all required columns are present
        for col in required_columns:
            assert col in results.columns
            
        # Check for lookahead bias in volume MA
        assert results['vol_ma'].iloc[0] != results['Volume'].iloc[0]
    
    def test_get_trade_summary(self, sample_market_data):
        analyzer = BreakoutAnalyzer()
        results = analyzer.calculate_signals(sample_market_data)
        summary = analyzer.get_trade_summary(results)
        
        required_metrics = [
            'total_trades', 'win_rate', 'avg_return', 'sharpe_ratio'
        ]
        
        # Check if all required metrics are present
        for metric in required_metrics:
            assert metric in summary
            
        # Check metric values are within reasonable ranges
        assert summary['total_trades'] >= 0
        assert 0 <= summary['win_rate'] <= 1
        assert isinstance(summary['avg_return'], float)
        assert isinstance(summary['sharpe_ratio'], float)
    
    def test_lookahead_bias(self, sample_market_data):
        """Test that the strategy doesn't use future data."""
        analyzer = BreakoutAnalyzer(volume_ma_period=3)  # Smaller period for easier testing
        results = analyzer.calculate_signals(sample_market_data)
        
        # Test volume MA calculation
        for i in range(len(results)):
            if i < 3:  # First 4 days should have NaN due to shift(1) and rolling(3)
                assert pd.isna(results['vol_ma'].iloc[i])
            else:
                # Check that MA only uses previous days' data
                expected_ma = sample_market_data['Volume'].iloc[i-3:i].mean()  # -4 to -1 due to shift(1)
                expectation = np.isclose(results['vol_ma'].iloc[i], expected_ma, rtol=1e-10)
                assert expectation
        
        # Test that strategy returns don't use future price data
        breakout_days = results[results['is_breakout']].index
        for day in breakout_days:
            day_idx = results.index.get_loc(day)
            # Entry price should be current day's close
            assert results.loc[day, 'entry_price'] == results.loc[day, 'Close']
            # Exit price should be N days later (where N is holding_period)
            if day_idx + analyzer.holding_period < len(results):
                future_day = results.index[day_idx + analyzer.holding_period]
                assert results.loc[day, 'exit_price'] == results.loc[future_day, 'Close']

def test_end_to_end(sample_market_data):
    """Test the entire workflow from data acquisition to analysis."""
    # Initialize components
    market_data = MarketData()
    analyzer = BreakoutAnalyzer(
        volume_ma_period=10,
        volume_threshold=1.5,
        price_threshold=0.01,
        holding_period=5
    )
    
    # Process data
    results = analyzer.calculate_signals(sample_market_data)
    summary = analyzer.get_trade_summary(results, ticker='AAPL')
    
    # Verify results
    assert isinstance(results, pd.DataFrame)
    assert isinstance(summary, dict)
    assert all(key in summary for key in [
        'total_trades', 'win_rate', 'avg_return', 'sharpe_ratio'
    ]) 