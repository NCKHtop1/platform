import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.signal import find_peaks
import plotly.graph_objects as go
import vectorbt as vbt
import pandas_ta as ta

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {color: #fff; background-color: #4CAF50; border-radius: 10px; border: none;}
    .stSidebar {background-color: #f0f2f6;}
    .css-1aumxhk {padding: 2rem;}
    </style>
    """, unsafe_allow_html=True)

# Add image to the landing page
st.image('image.png', use_column_width=True)

# Sector files mapping
SECTOR_FILES = {
    'Banking': 'Banking.csv',
    'Building Material': 'Building Material.csv',
    'Chemical': 'Chemical.csv',
    'Financial Services': 'Financial Services.csv',
    'Food and Beverage': 'Food and Beverage.csv',
    'Industrial Services': 'Industrial Services.csv',
    'Information Technology': 'Information Technology.csv',
    'Mineral': 'Mineral.csv',
    'Oil and Gas': 'Oil and Gas.csv',
    'Real Estate': 'Real Estate.csv',
    'Vnindex': 'Vnindex.csv'
}

# Load the dataset with conditional date parsing
@st.cache_data
def load_data(sector):
    file_path = SECTOR_FILES[sector]
    if sector == 'Vnindex':
        df = pd.read_csv(file_path)
        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%m/%d/%Y')  # Format for Vnindex
    else:
        df = pd.read_csv(file_path)
        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y', dayfirst=True)
    df.set_index('Datetime', inplace=True)
    return df

# Load unique stock symbols
@st.cache_data
def load_stock_symbols(sector):
    file_path = SECTOR_FILES[sector]
    df = pd.read_csv(file_path)
    stock_symbols_df = df.drop_duplicates(subset='StockSymbol')
    return stock_symbols_df['StockSymbol'].tolist()

# Ichimoku Oscillator Class
class IchimokuOscillator:
    def __init__(self, conversion_periods=8, base_periods=13, lagging_span2_periods=26, displacement=13):
        self.conversion_periods = conversion_periods
        self.base_periods = base_periods
        self.lagging_span2_periods = lagging_span2_periods
        self.displacement = displacement

    def donchian_channel(self, series, length):
        lowest = series.rolling(window=length, min_periods=1).min()
        highest = series.rolling(window=length, min_periods=1).max()
        return (lowest + highest) / 2

    def calculate(self, df):
        df['conversion_line'] = self.donchian_channel(df['close'], self.conversion_periods)
        df['base_line'] = self.donchian_channel(df['close'], self.base_periods)
        df['leading_span_a'] = (df['conversion_line'] + df['base_line']) / 2
        df['leading_span_b'] = self.donchian_channel(df['close'], self.lagging_span2_periods)
        df['cloud_min'] = np.minimum(df['leading_span_a'].shift(self.displacement - 1), df['leading_span_b'].shift(self.displacement - 1))
        df['cloud_max'] = np.maximum(df['leading_span_a'].shift(self.displacement - 1), df['leading_span_b'].shift(self.displacement - 1))
        return df

# Function to calculate MACD signals
def calculate_macd(prices, fast_length=12, slow_length=26, signal_length=9):
    def ema(values, length):
        alpha = 2 / (length + 1)
        ema_values = np.zeros_like(values)
        ema_values[0] = values[0]
        for i in range(1, len(values)):
            ema_values[i] = values[i] * alpha + ema_values[i - 1] * (1 - alpha)
        return ema_values

    fast_ma = ema(prices, fast_length)
    slow_ma = ema(prices, slow_length)
    macd_line = fast_ma - slow_ma

    signal_line = ema(macd_line, signal_length)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram

# Function to calculate buy/sell signals and crashes
def calculate_indicators_and_crashes(df, strategies):
    if "MACD" in strategies:
        macd = df.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)
        df['MACD Line'] = macd['MACD_12_26_9']
        df['Signal Line'] = macd['MACDs_12_26_9']
        df['MACD Buy'] = (df['MACD Line'] > df['Signal Line']) & (df['MACD Line'].shift(1) <= df['Signal Line'].shift(1))
        df['MACD Sell'] = (df['MACD Line'] < df['Signal Line']) & (df['MACD Line'].shift(1) >= df['Signal Line'].shift(1))

    if "Supertrend" in strategies:
        supertrend = df.ta.supertrend(length=7, multiplier=3, append=True)
        df['Supertrend'] = supertrend['SUPERTd_7_3.0']
        df['Supertrend Buy'] = supertrend['SUPERTd_7_3.0'] == 1  # Buy when supertrend is positive
        df['Supertrend Sell'] = supertrend['SUPERTd_7_3.0'] == -1  # Sell when supertrend is negative

    if "Stochastic" in strategies:
        stochastic = df.ta.stoch(append=True)
        df['Stochastic K'] = stochastic['STOCHk_14_3_3']
        df['Stochastic D'] = stochastic['STOCHd_14_3_3']
        df['Stochastic Buy'] = (df['Stochastic K'] > df['Stochastic D']) & (df['Stochastic K'].shift(1) <= df['Stochastic D'].shift(1))
        df['Stochastic Sell'] = (df['Stochastic K'] < df['Stochastic D']) & (df['Stochastic K'].shift(1) >= df['Stochastic D'].shift(1))

    if "RSI" in strategies:
        rsi = df.ta.rsi(close='close', length=14, append=True)
        df['RSI'] = rsi
        df['RSI Buy'] = df['RSI'] < 30  # RSI below 30 often considered as oversold
        df['RSI Sell'] = df['RSI'] > 70  # RSI above 70 often considered as overbought

    peaks, _ = find_peaks(df['close'])
    df['Peaks'] = df.index.isin(df.index[peaks])

    # Forward-fill peak prices to compute drawdowns
    peak_prices = df['close'].where(df['Peaks']).ffill()
    drawdowns = (peak_prices - df['close']) / peak_prices

    # Mark significant drawdowns as crashes
    crash_threshold = 0.175
    df['Crash'] = drawdowns >= crash_threshold

    # Adjust buy and sell signals based on crashes
    df['Adjusted Sell'] = ((df.get('MACD Sell', False) | df.get('Supertrend Sell', False) | df.get('Stochastic Sell', False) | df.get('RSI Sell', False)) &
                            (~df['Crash'].shift(1).fillna(False)))
    df['Adjusted Buy'] = ((df.get('MACD Buy', False) | df.get('Supertrend Buy', False) | df.get('Stochastic Buy', False) | df.get('RSI Buy', False)) &
                           (~df['Crash'].shift(1).fillna(False)))
    return df

# Function to run backtesting using vectorbt's from_signals
def run_backtest(df, init_cash, fees, direction):
    entries = df['Adjusted Buy']
    exits = df['Adjusted Sell']

    portfolio = vbt.Portfolio.from_signals(
        df['close'],
        entries,
        exits,
        init_cash=init_cash,
        fees=fees,
        direction=direction
    )
    return portfolio

## Streamlit App
st.title('Backtesting Stock and Index with Early Warning Model')
st.write('This application analyzes stocks with buy/sell signals and early warning signals of stock before the market crashes on HOSE and VNINDEX.')

# Sidebar: Sector selection
selected_sector = st.sidebar.selectbox('Select Sector', list(SECTOR_FILES.keys()))

# Load stock symbols and filter data
df_full = load_data(selected_sector)
stock_symbols = load_stock_symbols(selected_sector)
selected_stock_symbol = st.sidebar.selectbox('Select Stock Symbol', stock_symbols)

# Sidebar: Backtesting parameters
st.sidebar.header('Backtesting Parameters')
init_cash = st.sidebar.number_input('Initial Cash ($):', min_value=1000, max_value=1_000_000, value=100_000, step=1000)
fees = st.sidebar.number_input('Transaction Fees (%):', min_value=0.0, max_value=10.0, value=0.1, step=0.01) / 100
direction = st.sidebar.selectbox("Direction", ["longonly", "shortonly", "both"], index=0)
t_plus = st.sidebar.selectbox("T+ Settlement Days", [0, 1, 2.5, 3], index=0)  # Adding the T+ selection

# New trading parameters
take_profit_percentage = st.sidebar.number_input('Take Profit (%)', min_value=0.0, max_value=100.0, value=10.0, step=0.1)
stop_loss_percentage = st.sidebar.number_input('Stop Loss (%)', min_value=0.0, max_value=100.0, value=5.0, step=0.1)
trailing_take_profit_percentage = st.sidebar.number_input('Trailing Take Profit (%)', min_value=0.0, max_value=100.0, value=2.0, step=0.1)
trailing_stop_loss_percentage = st.sidebar.number_input('Trailing Stop Loss (%)', min_value=0.0, max_value=100.0, value=1.5, step=0.1)

# Sidebar: Choose the strategies to apply
strategies = st.sidebar.multiselect("Select Strategies", ["MACD", "Supertrend", "Stochastic", "RSI"], default=["MACD", "Supertrend", "Stochastic", "RSI"])

# Filter data for the selected stock symbol
symbol_data = df_full[df_full['StockSymbol'] == selected_stock_symbol]
symbol_data.sort_index(inplace=True)

# Automatically set the start date to the earliest available date for the selected symbol
first_available_date = symbol_data.index.min()
default_start_date = first_available_date.date() if first_available_date is not None else datetime(2000, 1, 1).date()
start_date = st.sidebar.date_input('Start Date', default_start_date)
end_date = st.sidebar.date_input('End Date', datetime.today().date())

if start_date < end_date:
    symbol_data = symbol_data.loc[start_date:end_date]

    # Calculate MACD, Ichimoku, and crash signals
    symbol_data = calculate_indicators_and_crashes(symbol_data, strategies)

    # Run backtest
    portfolio = run_backtest(symbol_data, init_cash, fees, direction)

    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Backtesting Stats", "List of Trades", "Equity Curve", "Drawdown", "Portfolio Plot"])

    with tab1:
        st.markdown("**Backtesting Stats:**")
        st.markdown("This tab displays the overall performance of the selected trading strategy. \
                    You'll find key metrics such as total return, profit/loss, and other relevant statistics.")
        stats_df = pd.DataFrame(portfolio.stats(), columns=['Value'])
        stats_df.index.name = 'Metric'
        st.dataframe(stats_df, height=800)

    with tab2:
        st.markdown("**List of Trades:**")
        st.markdown("This tab provides a detailed list of all trades executed by the strategy. \
                    You can analyze the entry and exit points of each trade, along with the profit or loss incurred.")
        trades_df = portfolio.trades.records_readable
        trades_df = trades_df.round(2)
        trades_df.index.name = 'Trade No'
        trades_df.drop(trades_df.columns[[0, 1]], axis=1, inplace=True)
        st.dataframe(trades_df, width=800, height=600)

    equity_data = portfolio.value()
    drawdown_data = portfolio.drawdown() * 100

    with tab3:
        equity_trace = go.Scatter(x=equity_data.index, y=equity_data, mode='lines', name='Equity', line=dict(color='green'))
        equity_fig = go.Figure(data=[equity_trace])
        equity_fig.update_layout(
            title='Equity Curve',
            xaxis_title='Date',
            yaxis_title='Equity',
            width=800,
            height=600
        )
        st.plotly_chart(equity_fig)
        st.markdown("**Equity Curve:**")
        st.markdown("This chart visualizes the growth of your portfolio value over time, \
                    allowing you to see how the strategy performs in different market conditions.")

    with tab4:
        drawdown_trace = go.Scatter(
            x=drawdown_data.index,
            y=drawdown_data,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='red')
        )
        drawdown_fig = go.Figure(data=[drawdown_trace])
        drawdown_fig.update_layout(
            title='Drawdown Curve',
            xaxis_title='Date',
            yaxis_title='% Drawdown',
            template='plotly_white',
            width=800,
            height=600
        )
        st.plotly_chart(drawdown_fig)
        st.markdown("**Drawdown Curve:**")
        st.markdown("This chart illustrates the peak-to-trough decline of your portfolio, \
                    giving you insights into the strategy's potential for losses.")

    with tab5:
        fig = portfolio.plot()
        crash_df = symbol_data[symbol_data['Crash']]
        fig.add_scatter(
            x=crash_df.index,
            y=crash_df['close'],
            mode='markers',
            marker=dict(color='orange', size=10, symbol='triangle-down'),
            name='Crash'
        )
        st.markdown("**Portfolio Plot:**")
        st.markdown("This comprehensive plot combines the equity curve with buy/sell signals and potential crash warnings, \
                    providing a holistic view of the strategy's performance.")
        st.plotly_chart(fig, use_container_width=True)

# If the end date is before the start date, show an error
if start_date > end_date:
    st.error('Error: End Date must fall after Start Date.')

else:
    st.write("Please select a valid date range to view the results.")
