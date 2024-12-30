# Breakout thesis

## Description

If a stock is getting traded twice as the daily average and its up 2% on that day compared to the prior day, we should buy it and hold it for 10 days to see what happens

## Example usage

```[python]
    from breakout_thesis.app import MarketData, BreakoutAnalyzer
    from datetime import date

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
        summary = analyzer.get_trade_summary(results)
        print(f"Strategy Results for {ticker}:")
        print(summary)
```

## Results
