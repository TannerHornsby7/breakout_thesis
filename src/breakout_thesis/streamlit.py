import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
import numpy as np
from app import MarketData, BreakoutAnalyzer

def plot_strategy_results(df: pd.DataFrame, ticker: str):
    """ An interactive plot showing price, volume, and signals."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{ticker} Price and Signals', 'Volume'),
        row_heights=[0.7, 0.3]
    )

    # Price chart with signals
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )

    # Add breakout points
    breakout_days = df[df['is_breakout']].index
    fig.add_trace(
        go.Scatter(
            x=breakout_days,
            y=df.loc[breakout_days, 'Close'],
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=10,
                color='green',
            ),
            name='Breakout Signals'
        ),
        row=1, col=1
    )

    # Volume chart
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color='rgba(0,0,255,0.5)'
        ),
        row=2, col=1
    )

    # Add volume MA
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['vol_ma'],
            name='Volume MA',
            line=dict(color='red', width=1)
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False,
        template='plotly_white'
    )

    return fig

def plot_returns_distribution(returns: pd.Series):
    """Create a histogram of strategy returns."""
    fig = go.Figure()
    
    fig.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=30,
            name='Returns Distribution',
            marker_color='rgba(0,0,255,0.5)'
        )
    )

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="red",
        annotation_text="Break Even"
    )

    fig.update_layout(
        title='Distribution of Strategy Returns',
        xaxis_title='Return (%)',
        yaxis_title='Frequency',
        template='plotly_white'
    )

    return fig

def create_metrics_cards(summary: dict):
    """Display strategy metrics in a clean format."""
    cols = st.columns(4)
    
    # Total Trades
    with cols[0]:
        st.metric(
            "Total Trades",
            f"{summary['total_trades']}",
            help="Number of breakout signals generated"
        )
    
    # Win Rate
    with cols[1]:
        st.metric(
            "Win Rate",
            f"{summary['win_rate']:.1%}",
            help="Percentage of trades with positive returns"
        )
    
    # Average Return
    with cols[2]:
        st.metric(
            "Avg Return",
            f"{summary['avg_return']:.1%}",
            help="Mean return per trade"
        )
    
    # Sharpe Ratio
    with cols[3]:
        st.metric(
            "Sharpe Ratio",
            f"{summary['sharpe_ratio']:.2f}",
            help="Risk-adjusted return metric (annualized)"
        )

def main():
    st.set_page_config(
        page_title="Volume Breakout Strategy Analyzer",
        page_icon="📈",
        layout="wide"
    )

    st.title("📈 Volume Breakout Strategy Analyzer")
    st.markdown("""
    This tool analyzes a volume breakout strategy across different stocks and parameters.
    The strategy looks for days with significant volume and price increases as potential
    entry signals.
    """)

    # Sidebar inputs
    st.sidebar.header("Strategy Parameters")
    
    # Ticker input
    ticker = st.sidebar.text_input(
        "Stock Ticker",
        value="AAPL",
        help="Enter a valid stock ticker symbol"
    ).upper()

    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            date.today() - timedelta(days=365),
            help="Analysis start date"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            date.today(),
            help="Analysis end date"
        )

    # Strategy parameters
    volume_ma_period = st.sidebar.slider(
        "Volume MA Period",
        min_value=5,
        max_value=50,
        value=20,
        help="Number of days for volume moving average"
    )

    volume_threshold = st.sidebar.slider(
        "Volume Threshold",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.1,
        help="Multiple of average volume required for breakout"
    )

    price_threshold = st.sidebar.slider(
        "Price Change Threshold",
        min_value=0.0,
        max_value=0.10,
        value=0.02,
        step=0.005,
        format="%.1f%%",
        help="Minimum price increase required for breakout"
    )

    holding_period = st.sidebar.slider(
        "Holding Period",
        min_value=1,
        max_value=30,
        value=10,
        help="Number of days to hold position after breakout"
    )

    # Initialize classes
    market_data = MarketData()
    analyzer = BreakoutAnalyzer(
        volume_ma_period=volume_ma_period,
        volume_threshold=volume_threshold,
        price_threshold=price_threshold,
        holding_period=holding_period
    )

    # Get and process data
    with st.spinner('Fetching and analyzing data...'):
        df = market_data.get_historical_data(ticker, start_date, end_date)
        
        if df is None:
            st.error(f"Could not fetch data for {ticker}. Please check the ticker symbol.")
            return

        results = analyzer.calculate_signals(df)
        summary = analyzer.get_trade_summary(results, ticker)

    # Display results
    st.subheader("Strategy Performance Metrics")
    create_metrics_cards(summary)

    # Charts
    st.subheader("Price and Volume Analysis")
    fig1 = plot_strategy_results(results, ticker)
    st.plotly_chart(fig1, use_container_width=True)

    # Returns distribution
    if summary['total_trades'] > 0:
        st.subheader("Returns Distribution")
        returns = results[results['is_breakout']]['strategy_return'] * 100  # Convert to percentage
        fig2 = plot_returns_distribution(returns)
        st.plotly_chart(fig2, use_container_width=True)

    # Trade list
    st.subheader("Individual Trades")
    trades = results[results['is_breakout']].copy()
    if len(trades) > 0:
        trade_data = pd.DataFrame({
            'Entry Date': trades.index,
            'Entry Price': trades['entry_price'].round(2),
            'Exit Price': trades['exit_price'].round(2),
            'Return (%)': (trades['strategy_return'] * 100).round(2)
        })
        st.dataframe(
            trade_data.style.background_gradient(
                subset=['Return (%)'],
                cmap='RdYlGn',
                vmin=-10,
                vmax=10
            ),
            hide_index=True
        )
    else:
        st.info("No trades were generated with the current parameters.")

if __name__ == "__main__":
    main() 