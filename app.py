import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import vectorbt as vbt
import pandas_ta as ta
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

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

@st.cache(allow_output_mutation=True)
def load_data(sector):
    file_path = SECTOR_FILES[sector]
    df = pd.read_csv(file_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%m/%d/%Y' if sector == 'Vnindex' else '%d/%m/%Y', dayfirst=True if sector != 'Vnindex' else False)
    df.set_index('Datetime', inplace=True)
    df['normalized_close'] = (df['close'] - df['close'].mean()) / df['close'].std()
    return df

def create_and_train_vae(df):
    x_train = df['normalized_close'].values.reshape(-1, 1)
    vae = VAE(latent_dim=2, input_shape=(1,))
    vae.compile(optimizer=optimizers.Adam(learning_rate=0.001))
    vae.fit(x_train, epochs=50, batch_size=32)
    return vae

def detect_anomalies(df, vae):
    x_train = df['normalized_close'].values.reshape(-1, 1)
    reconstructed, _, _ = vae.predict(x_train)
    reconstruction_error = np.abs(x_train - reconstructed.flatten())
    threshold = np.mean(reconstruction_error) + 2 * np.std(reconstruction_error)
    df['Anomaly'] = reconstruction_error > threshold
    return df

# Streamlit application setup
st.title('Backtesting Stock and Index with Early Warning Model')

# Sidebar configuration
selected_sector = st.sidebar.selectbox('Select Sector', list(SECTOR_FILES.keys()))
df_full = load_data(selected_sector)

if 'vae' not in st.session_state:
    st.session_state.vae = create_and_train_vae(df_full)

df_full = detect_anomalies(df_full, st.session_state.vae)

init_cash = st.sidebar.number_input('Initial Cash ($):', min_value=1000, max_value=1_000_000, value=100_000, step=1000)
fees = st.sidebar.number_input('Transaction Fees (%):', min_value=0.0, max_value=10.0, value=0.1, step=0.01) / 100
direction = st.sidebar.selectbox("Direction", ["longonly", "shortonly", "both"], index=0)

stock_symbols = df_full['StockSymbol'].unique()
selected_stock_symbol = st.sidebar.selectbox('Select Stock Symbol', stock_symbols)

# Filtering data for the selected stock symbol
df_selected = df_full[df_full['StockSymbol'] == selected_stock_symbol]
df_selected.sort_index(inplace=True)

start_date = st.sidebar.date_input('Start Date', df_selected.index.min())
end_date = st.sidebar.date_input('End Date', datetime.today().date())

# Processing data
if start_date < end_date:
    df_selected = df_selected.loc[start_date:end_date]

    portfolio = vbt.Portfolio.from_signals(
        df_selected['close'], entries=df_selected['Anomaly'], exits=df_selected['Anomaly'],
        init_cash=init_cash, freq='1D', fees=fees
    )

# Create tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Backtesting Stats", "List of Trades", "Equity Curve", "Drawdown", "Portfolio Plot"])

with tab1:
    st.write(portfolio.stats())

with tab2:
    trades_df = portfolio.trades.records_readable
    st.dataframe(trades_df)

with tab3:
    equity_curve = go.Figure(data=[go.Scatter(x=df_selected.index, y=portfolio.equity(), mode='lines')])
    equity_curve.update_layout(title='Equity Curve', xaxis_title='Date', yaxis_title='Equity Value')
    st.plotly_chart(equity_curve, use_container_width=True)

with tab4:
    drawdown_data = portfolio.drawdown() * 100
    drawdown_curve = go.Figure(data=[go.Scatter(x=df_selected.index, y=drawdown_data, mode='lines', fill='tozeroy')])
    drawdown_curve.update_layout(title='Drawdown Curve', xaxis_title='Date', yaxis_title='% Drawdown')
    st.plotly_chart(drawdown_curve, use_container_width=True)

with tab5:
    fig = go.Figure(data=[go.Scatter(x=df_selected.index, y=df_selected['close'], mode='lines', name='Close'),
                          go.Scatter(x=df_selected.index[df_selected['Anomaly']], y=df_selected['close'][df_selected['Anomaly']], mode='markers', marker=dict(color='red', size=10), name='Anomalies')])
    fig.update_layout(title='Portfolio Plot with Anomalies', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True)

# Handle incorrect date range selection
if start_date > end_date:
    st.error('Error: End Date must fall after Start Date.')
