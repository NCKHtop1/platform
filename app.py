import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import vectorbt as vbt
import pandas_ta as ta
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers

# Custom Sampling layer for VAE
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Variational Autoencoder (VAE) model
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
            reconstruction_loss = tf.reduce_mean(losses.mean_squared_error(x, reconstructed))
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
            loss = reconstruction_loss + tf.reduce_mean(kl_loss)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {'loss': loss}

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

# Load data
@st.cache(allow_output_mutation=True)
def load_data(sector):
    df = pd.read_csv(SECTOR_FILES[sector])
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y', dayfirst=True)
    df.set_index('Datetime', inplace=True)
    df['normalized_close'] = (df['close'] - df['close'].mean()) / df['close'].std()
    return df

# Detect anomalies using VAE
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

# Main app setup
st.title('Backtesting Stock and Index with Early Warning Model')
st.write('This application analyzes stocks with buy/sell signals, Early warning signals of stock before the market crashes on HOSE and VNINDEX.')

# Sidebar: Sector selection
selected_sector = st.sidebar.selectbox('Select Sector', list(SECTOR_FILES.keys()))

# Load and detect anomalies
df_full = load_data(selected_sector)
df_full = detect_anomalies(df_full)

# Sidebar: Backtesting parameters
st.sidebar.header('Backtesting Parameters')
init_cash = st.sidebar.number_input('Initial Cash ($):', min_value=1000, max_value=1_000_000, value=100000, step=1000)
fees = st.sidebar.number_input('Transaction Fees (%):', min_value=0.0, max_value=10.0, value=0.1, step=0.01) / 100
direction = st.sidebar.selectbox("Direction", ["longonly", "shortonly", "both"], index=0)

# Load unique stock symbols and filter data
stock_symbols_df = df_full.drop_duplicates(subset='StockSymbol')
stock_symbols = stock_symbols_df['StockSymbol'].tolist()
selected_stock_symbol = st.sidebar.selectbox('Select Stock Symbol', stock_symbols)

# Filter data for the selected stock symbol
symbol_data = df_full[df_full['StockSymbol'] == selected_stock_symbol]
symbol_data.sort_index(inplace=True)

# Sidebar: Date input
default_start_date = symbol_data.index.min().date() if symbol_data.index.min() is not None else datetime(2000, 1, 1).date()
start_date = st.sidebar.date_input('Start Date', default_start_date)
end_date = st.sidebar.date_input('End Date', datetime.today().date())

# Filter by date range
if start_date < end_date:
    symbol_data = symbol_data.loc[start_date:end_date]

    # Calculate indicators and run backtest
    strategies = ['MACD', 'Supertrend', 'Stochastic', 'RSI']  # Define your strategies
    symbol_data = calculate_indicators_and_crashes(symbol_data, strategies)  # This function needs to be defined or modified according to your strategy calculation
    portfolio = run_backtest(symbol_data, init_cash, fees, direction)  # This function needs to be defined or modified according to your backtesting logic

    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Backtesting Stats", "List of Trades", "Equity Curve", "Drawdown", "Portfolio Plot"])

    with tab1:
        st.markdown("**Backtesting Stats:**")
        stats_df = pd.DataFrame(portfolio.stats(), columns=['Value'])
        stats_df.index.name = 'Metric'
        st.dataframe(stats_df, height=800)

    with tab2:
        st.markdown("**List of Trades:**")
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
        equity_fig.update_layout(title='Equity Curve', xaxis_title='Date', yaxis_title='Equity', width=800, height=600)
        st.plotly_chart(equity_fig)

    with tab4:
        drawdown_trace = go.Scatter(x=drawdown_data.index, y=drawdown_data, mode='lines', name='Drawdown', fill='tozeroy', line=dict(color='red'))
        drawdown_fig = go.Figure(data=[drawdown_trace])
        drawdown_fig.update_layout(title='Drawdown Curve', xaxis_title='Date', yaxis_title='% Drawdown', template='plotly_white', width=800, height=600)
        st.plotly_chart(drawdown_fig)

    with tab5:
        fig = portfolio.plot()
        crash_df = symbol_data[symbol_data['Anomaly']]
        fig.add_scatter(x=crash_df.index, y=crash_df['close'], mode='markers', marker=dict(color='orange', size=10, symbol='triangle-down'), name='Crash')
        st.plotly_chart(fig, use_container_width=True)

# Handle incorrect date range selection
if start_date > end_date:
    st.error('Error: End Date must fall after Start Date.')
