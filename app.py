import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.signal import find_peaks
import plotly.graph_objects as go
import vectorbt as vbt
import pandas_ta as ta
from scipy.stats import chi2
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import MeanSquaredError

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

    # Identify crashes using VAE anomaly detection
    df['Crash'] = identify_anomalies(df['close'])

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

# Custom Sampling layer for VAE
class VAE(models.Model):
    def __init__(self, latent_dim, input_shape):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = models.Sequential([
            layers.InputLayer(input_shape=input_shape),
            layers.Dense(128, activation='relu'),
            layers.Dense(latent_dim * 2),  # Outputs for mean and log variance
        ])
        self.decoder = models.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(input_shape[0]),  # Output same as input shape
        ])
        self.sampling = Sampling()

    def encode(self, x):
        z_mean_log_var = self.encoder(x)
        z_mean, z_log_var = tf.split(z_mean_log_var, num_or_size_splits=2, axis=1)
        return z_mean, z_log_var

    def decode(self, z):
        logits = self.decoder(z)
        return logits

    def call(self, inputs):
        z_mean, z_log_var = self.encode(inputs)
        z = self.sampling((z_mean, z_log_var))
        reconstructed = self.decode(z)
        return reconstructed

    def compute_loss(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.sampling((z_mean, z_log_var))
        x_logit = self.decode(z)

        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.mse(x, x_logit))
        kl_loss = -0.5 * tf.reduce_sum(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1, axis=1)
        total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        return total_loss

# Anomaly detection using VAE
def identify_anomalies(series):
    # Normalize data
    series = (series - series.min()) / (series.max() - series.min())

    # Prepare data for VAE
    series = series.values.reshape(-1, 1).astype('float32')
    input_shape = series.shape[1:]
    latent_dim = 2

    vae = VAE(latent_dim, input_shape)
    vae.compile(optimizer=optimizers.Adam())

    # Custom training loop to use `compute_loss`
    @tf.function
    def train_step(x):
        with tf.GradientTape() as tape:
            loss = vae.compute_loss(x)
        gradients = tape.gradient(loss, vae.trainable_variables)
        vae.optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
        return loss

    # Train VAE
    for epoch in range(50):
        train_step(series)

    # Get reconstruction errors
    reconstructed = vae.predict(series)
    reconstruction_errors = np.mean(np.square(series - reconstructed), axis=1)

    # Chi-squared test for anomaly detection
    threshold = chi2.ppf(0.99, df=latent_dim)
    anomalies = reconstruction_errors > threshold

    return anomalies

# Streamlit App
st.set_page_config(page_title="Stock Market Analysis App", layout="wide")

st.title("Stock Market Analysis App")

sector = st.sidebar.selectbox("Select Sector", list(SECTOR_FILES.keys()))

df = load_data(sector)
stock_symbols = load_stock_symbols(sector)

stock = st.sidebar.selectbox("Select Stock", stock_symbols)

strategies = st.sidebar.multiselect(
    "Select Strategies",
    ["MACD", "Supertrend", "Stochastic", "RSI"],
    default=["MACD", "Supertrend", "Stochastic", "RSI"]
)

df_stock = df[df['StockSymbol'] == stock]

ichimoku = IchimokuOscillator()
df_stock = ichimoku.calculate(df_stock)

df_stock = calculate_indicators_and_crashes(df_stock, strategies)

initial_cash = st.sidebar.number_input("Initial Cash", min_value=1000, max_value=1000000, value=10000)
fees = st.sidebar.number_input("Fees (%)", min_value=0.0, max_value=1.0, value=0.1) / 100
direction = st.sidebar.selectbox("Trade Direction", ["longonly", "shortonly", "both"], index=0)

portfolio = run_backtest(df_stock, initial_cash, fees, direction)

st.subheader(f"{stock} Stock Data")
st.write(df_stock)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['close'], mode='lines', name='Close'))
fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['cloud_min'], mode='lines', name='Cloud Min', line=dict(dash='dot')))
fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['cloud_max'], mode='lines', name='Cloud Max', line=dict(dash='dot')))

buy_signals = df_stock[df_stock['Adjusted Buy']]
sell_signals = df_stock[df_stock['Adjusted Sell']]

fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['close'], mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', color='green', size=10)))
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['close'], mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', color='red', size=10)))

st.plotly_chart(fig)

st.subheader("Backtest Results")
st.write(portfolio.stats())

fig2 = portfolio.plot().update_layout(title_text='Portfolio Performance')
st.plotly_chart(fig2)
