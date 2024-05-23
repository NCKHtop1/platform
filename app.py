import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import vectorbt as vbt
import pandas_ta as ta
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers

# Define the custom Sampling layer and VAE model
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(models.Model):
    def __init__(self, latent_dim, input_shape):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = models.Sequential([
            layers.InputLayer(input_shape=input_shape),
            layers.Dense(128, activation='relu'),
            layers.Dense(latent_dim * 2),
        ])
        self.decoder = models.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(input_shape[0], activation='sigmoid'),
        ])
        self.sampling = Sampling()

    def encode(self, x):
        z_mean, z_log_var = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        return self.sampling((z_mean, z_log_var))

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decode(z)
        return reconstructed, z_mean, z_log_var

    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            reconstructed, z_mean, z_log_var = self(x, training=True)
            mse = tf.keras.losses.MeanSquaredError()
            reconstruction_loss = mse(x, reconstructed)
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {'loss': total_loss}

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def detect_anomalies(df):
    x_train = df['normalized_close'].values.reshape(-1, 1)
    vae = VAE(latent_dim=2, input_shape=(1,))
    vae.compile(optimizer=optimizers.Adam(learning_rate=0.001))
    # Avoid training inside the cached function
    if not hasattr(st, 'vae'):
        st.vae = vae
        st.vae.fit(x_train, epochs=50, batch_size=32)
    reconstructed, _, _ = st.vae.predict(x_train)
    reconstruction_error = np.abs(x_train - reconstructed.flatten())  # Ensure flattening if necessary
    threshold = np.mean(reconstruction_error) + 2 * np.std(reconstruction_error)
    df['Anomaly'] = reconstruction_error > threshold
    return df

# Configurations and File Paths
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

# Load data and normalize
@st.cache(allow_output_mutation=True)
def load_data(sector):
    file_path = SECTOR_FILES[sector]
    df = pd.read_csv(file_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y', dayfirst=True)
    df.set_index('Datetime', inplace=True)
    df['normalized_close'] = (df['close'] - df['close'].mean()) / df['close'].std()
    return df

# Anomaly detection with VAE
@st.cache(allow_output_mutation=True)
def detect_anomalies(df):
    x_train = df['normalized_close'].values.reshape(-1, 1)
    vae = VAE(latent_dim=2, input_shape=(1,))
    vae.compile(optimizer=optimizers.Adam(learning_rate=0.001))
    vae.fit(x_train, epochs=50, batch_size=32)
    reconstructed, _, _ = vae.predict(x_train)
    reconstruction_error = np.abs(x_train - reconstructed)
    threshold = np.mean(reconstruction_error) + 2 * np.std(reconstruction_error)
    df['Anomaly'] = reconstruction_error > threshold
    return df

# Calculate indicators and crashes (revised to incorporate anomaly detection)
def calculate_indicators_and_crashes(df, strategies):
    if "MACD" in strategies:
        df = df.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)
    if "RSI" in strategies:
        df = df.ta.rsi(close='close', length=14, append=True)

    # Adjust buy and sell signals based on detected anomalies
    df['Adjusted Buy'] = (df['MACD_Hist'] > 0) & (~df['Anomaly'])
    df['Adjusted Sell'] = (df['MACD_Hist'] < 0) & (~df['Anomaly'])
    return df

# Streamlit Application Layout
st.title('Stock and Index Backtesting with Anomaly Detection')
st.write('This application analyzes stocks with buy/sell signals, and includes early warning signals of potential crashes based on anomaly detection.')

# Sidebar for sector and stock selection
selected_sector = st.sidebar.selectbox('Select Sector', list(SECTOR_FILES.keys()))
df_full = load_data(selected_sector)
df_full = detect_anomalies(df_full)

stock_symbols = df_full['StockSymbol'].unique()
selected_stock_symbol = st.sidebar.selectbox('Select Stock Symbol', stock_symbols)

# Trading parameters
init_cash = st.sidebar.number_input('Initial Cash ($):', min_value=1000, max_value=1_000_000, value=100000, step=1000)
fees = st.sidebar.number_input('Transaction Fees (%):', min_value=0.0, max_value=10.0, value=0.1, step=0.01) / 100
strategies = st.sidebar.multiselect("Select Strategies", ["MACD", "RSI"], default=["MACD"])

# Select date range
df_selected = df_full[df_full['StockSymbol'] == selected_stock_symbol]
start_date = st.sidebar.date_input('Start Date', df_selected.index.min())
end_date = st.sidebar.date_input('End Date', df_selected.index.max())
df_selected = df_selected.loc[start_date:end_date]

# Calculate indicators and signals
df_selected = calculate_indicators_and_crashes(df_selected, strategies)

# Run backtesting
portfolio = vbt.Portfolio.from_signals(
    df_selected['close'], entries=df_selected['Adjusted Buy'], exits=df_selected['Adjusted Sell'],
    init_cash=init_cash, freq='1D', fees=fees
)

# Display Results in Tabs
tab1, tab2 = st.tabs(['Equity Curve', 'Signals and Anomalies'])
with tab1:
    st.header("Equity Curve")
    st.plotly_chart(vbt.plotting.create_returns_tear_sheet(portfolio, benchmark_rets=None), use_container_width=True)

with tab2:
    st.header("Signals and Anomalies")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_selected.index, y=df_selected['close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=df_selected.index[df_selected['Adjusted Buy']], y=df_selected['close'][df_selected['Adjusted Buy']], mode='markers', marker=dict(color='green', size=10), name='Buy Signal'))
    fig.add_trace(go.Scatter(x=df_selected.index[df_selected['Adjusted Sell']], y=df_selected['close'][df_selected['Adjusted Sell']], mode='markers', marker=dict(color='red', size=10), name='Sell Signal'))
    fig.add_trace(go.Scatter(x=df_selected.index[df_selected['Anomaly']], y=df_selected['close'][df_selected['Anomaly']], mode='markers', marker=dict(color='orange', size=10), name='Anomaly Detected'))
    fig.update_layout(title='Price with Trading Signals and Anomalies', xaxis_title='Date', yaxis_title='Price', legend_title='Legend')
    st.plotly_chart(fig, use_container_width=True)

# Error handling for date range
if start_date > end_date:
    st.error('Error: End Date must fall after Start Date.')
