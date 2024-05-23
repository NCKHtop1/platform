import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import vectorbt as vbt
from datetime import datetime
import pandas_ta as ta
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

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
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y', dayfirst=True)
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

st.title('Backtesting Stock and Index with Early Warning Model')
selected_sector = st.sidebar.selectbox('Select Sector', list(SECTOR_FILES.keys()))
df_full = load_data(selected_sector)

if 'vae' not in st.session_state or 'last_sector' not in st.session_state or st.session_state.last_sector != selected_sector:
    st.session_state.vae = create_and_train_vae(df_full)
    st.session_state.last_sector = selected_sector

df_full = detect_anomalies(df_full, st.session_state.vae)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Backtesting Stats", "List of Trades", "Equity Curve", "Drawdown", "Anomalies"])

with tab1:
    st.write('Backtesting statistics will go here.')

with tab2:
    st.write('List of trades will go here.')

with tab3:
    equity_curve = go.Figure(data=[go.Scatter(x=df_full.index, y=df_full['close'], mode='lines')])
    equity_curve.update_layout(title='Equity Curve', xaxis_title='Date', yaxis_title='Equity Value')
    st.plotly_chart(equity_curve)

with tab4:
    st.write('Drawdown information will go here.')

with tab5:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_full.index, y=df_full['close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=df_full.index[df_full['Anomaly']], y=df_full['close'][df_full['Anomaly']], mode='markers', marker=dict(color='red', size=10), name='Anomaly'))
    fig.update_layout(title="Detected Anomalies in Price Data", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

if 'vae' in st.session_state:
    st.success("Model trained and anomalies detected successfully!")
else:
    st.error("Failed to train the model due to an error in data.")
