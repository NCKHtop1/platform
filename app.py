import streamlit as st
import pandas as pd
import numpy as np
import os
from scipy.signal import find_peaks
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import vectorbt as vbt
import pandas_ta as ta
from vnstock import stock_historical_data

# Check if the image file exists
image_path = 'image.png'
if not os.path.exists(image_path):
    st.error(f"Image file not found: {image_path}")
else:
    st.image(image_path, use_column_width=True)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {color: #fff; background-color: #4CAF50; border-radius: 10px; border: none;}
    .stSidebar {background-color: #f0f2f6;}
    </style>
    """, unsafe_allow_html=True)

# Sector and Portfolio files mapping
SECTOR_FILES = {
    'Ngân hàng': 'Banking.csv',
    'Vật liệu xây dựng': 'Building Material.csv',
    'Hóa chất': 'Chemical.csv',
    'Dịch vụ tài chính': 'Financial Services.csv',
    'Thực phẩm và đồ uống': 'Food and Beverage.csv',
    'Dịch vụ công nghiệp': 'Industrial Services.csv',
    'Công nghệ thông tin': 'Information Technology.csv',
    'Khoáng sản': 'Mineral.csv',
    'Dầu khí': 'Oil and Gas.csv',
    'Bất động sản': 'Real Estate.csv',
    'VNINDEX': 'Vnindex.csv'
}

# Load data function
@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()
    return pd.read_csv(file_path, parse_dates=['Datetime'], dayfirst=True).set_index('Datetime')

def ensure_datetime_compatibility(start_date, end_date, df):
    df = df[~df.index.duplicated(keep='first')]  # Ensure unique indices
    if not isinstance(start_date, pd.Timestamp):
        start_date = pd.Timestamp(start_date)
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.Timestamp(end_date)

    # Check if the dates are within the dataframe's range
    if start_date not in df.index:
        start_date = df.index[df.index.searchsorted(start_date)]
    if end_date not in df.index:
        end_date = df.index[df.index.searchsorted(end_date)]

    return df.loc[start_date:end_date]

# Load and filter detailed data
def load_detailed_data(selected_stocks):
    data = pd.DataFrame()
    for sector, file_path in SECTOR_FILES.items():
        df = load_data(file_path)
        if not df.empty:
            sector_data = df[df['StockSymbol'].isin(selected_stocks)]
            data = pd.concat([data, sector_data])
    return data

def calculate_VaR(returns, confidence_level=0.95):
    """
    Calculate the Value at Risk (VaR) for a series of returns
    :param returns: pandas.Series of returns
    :param confidence_level: float, confidence level for VaR
    :return: float, VaR value
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    var = np.percentile(returns, 100 * (1 - confidence_level))
    return var

# Define the VN30 class
class VN30:
    def __init__(self):
        self.symbols = [
            "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
            "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
            "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"
        ]

    def fetch_data(self, symbol, start_date, end_date):
        data = stock_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            resolution='1D',
            type='stock',
            beautify=True,
            decor=False,
            source='DNSE'
        )
        df = pd.DataFrame(data)
        if not df.empty:
            df.rename(columns={'time': 'Datetime', 'ticker': 'StockSymbol'}, inplace=True)
            df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
            return df.set_index('Datetime', drop=True)
        return pd.DataFrame()  # Handle case where no data is returned

    def analyze_stocks(self, selected_symbols, start_date, end_date):
        results = []
        for symbol in selected_symbols:
            stock_data = self.fetch_data(symbol, start_date, end_date)
            if not stock_data.empty:
                stock_data['Crash Risk'] = self.calculate_crash_risk(stock_data)
                results.append(stock_data)
        if results:
            combined_data = pd.concat(results)
            return combined_data
        else:
            return pd.DataFrame()  # Handle case where no data is returned

    def calculate_crash_risk(self, df):
        df['returns'] = df['close'].pct_change()
        df['VaR'] = df.groupby('StockSymbol')['returns'].transform(lambda x: calculate_VaR(x))
        conditions = [
            (df['VaR'] < -0.02),  # Adjust these thresholds based on your risk tolerance
            (df['VaR'].between(-0.02, -0.01)),
            (df['VaR'] > -0.01)
        ]
        choices = ['High', 'Medium', 'Low']
        df['Crash Risk'] = np.select(conditions, choices, default='Low')
        return df

    def display_stock_status(self, df):
        if df.empty:
            st.error("No data available.")
            return
    
        if 'Crash Risk' not in df.columns or 'StockSymbol' not in df.columns:
            st.error("Data is missing necessary columns ('Crash Risk' or 'StockSymbol').")
            return
    
        color_map = {'Low': '#4CAF50', 'Medium': '#FFC107', 'High': '#FF5733'}
        n_cols = 5
        n_rows = (len(df) + n_cols - 1) // n_cols  # Determine the number of rows needed
    
        for i in range(n_rows):
            cols = st.columns(n_cols)  # Create a row of columns
            for j, col in enumerate(cols):
                idx = i * n_cols + j
                if idx < len(df):
                    data_row = df.iloc[idx]
                    crash_risk = data_row.get('Crash Risk', 'Unknown')  # Safely get the crash risk
                    stock_symbol = data_row['StockSymbol']  # Get the stock symbol
                    color = color_map.get(crash_risk, '#FF5722')  # Get the color for the crash risk
                    date = data_row.name.strftime('%Y-%m-%d')  # Format the date
    
                    # Display the colored box with the symbol, date, and crash risk
                    col.markdown(
                        f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; text-align: center;'>"
                        f"<strong>{stock_symbol}</strong><br>{date}<br>{crash_risk}</div>", 
                        unsafe_allow_html=True
                    )
                else:
                    col.empty()  

# Usage in Streamlit (main application flow)
st.title('Bảng Phân Tích Cổ Phiếu Trong Danh Mục VN30')
vn30 = VN30()
selected_symbols = vn30.symbols  # Assuming all symbols are selected for simplicity

# Sidebar for Portfolio Selection
with st.sidebar.expander("Danh mục đầu tư", expanded=True):
    vn30 = VN30()
    selected_stocks = []
    portfolio_options = st.multiselect('Chọn danh mục', ['VN30', 'Chọn mã theo ngành'], key='portfolio_options')

    display_vn30 = 'VN30' in portfolio_options  # Set to True only if VN30 is selected

    if 'VN30' in portfolio_options:
        selected_symbols = st.multiselect('Chọn mã cổ phiếu trong VN30', vn30.symbols, default=vn30.symbols, key='vn30_symbols')
        
    if 'Chọn mã theo ngành' in portfolio_options:
        selected_sector = st.selectbox('Chọn ngành để lấy dữ liệu', list(SECTOR_FILES.keys()), key='sector_selection')
        if selected_sector:
            df_full = load_data(SECTOR_FILES[selected_sector])
            available_symbols = df_full['StockSymbol'].unique().tolist()
            sector_selected_symbols = st.multiselect('Chọn mã cổ phiếu trong ngành', available_symbols, key='sector_symbols')
            selected_stocks.extend(sector_selected_symbols)
            display_vn30 = False  # Disable VN30 display if sector is selected

    # Display color key for crash risk
    st.markdown("""
    <div style='margin-top: 20px;'>
        <strong>Chỉ số Đánh Giá Rủi Ro Sụp Đổ:</strong>
        <ul>
            <li><span style='color: #FF5733;'>Màu Đỏ: Rủi Ro Cao</span></li>
            <li><span style='color: #FFC107;'>Màu Vàng: Rủi Ro Trung Bình</span></li>
            <li><span style='color: #4CAF50;'>Màu Xanh Lá: Rủi Ro Thấp</span></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if st.sidebar.button('Kết Quả', key='result_button'):
    vn30_stocks = vn30.analyze_stocks(selected_symbols, '2024-01-25', pd.Timestamp.today().strftime('%Y-%m-%d'))
    if not vn30_stocks.empty:
        st.write("Hiển thị kết quả sự sụt giảm cổ phiếu trong danh mục VN30 ngày hôm nay.")
        st.write("""
        <div>
            <strong>Chú thích màu sắc:</strong>
            <ul>
                <li><span style='color: #FF5733;'>Màu Đỏ: Rủi Ro Cao</span> - Rủi ro sụt giảm giá cao.</li>
                <li><span style='color: #FFC107;'>Màu Vàng: Rủi Ro Trung Bình</span> - Rủi ro sụt giảm giá trung bình.</li>
                <li><span style='color: #4CAF50;'>Màu Xanh Lá: Rủi Ro Thấp</span> - Rủi ro sụt giảm giá thấp.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        vn30.display_stock_status(vn30_stocks)
    else:
        st.error("Không có dữ liệu cho cổ phiếu VN30 hôm nay.")

def calculate_indicators_and_crashes(df, strategies):
    if df.empty:
        st.error("No data available for the selected date range.")
        return df

    try:
        if "MACD" in strategies:
            macd = df.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)
            if 'MACD_12_26_9' in macd.columns:
                df['MACD Line'] = macd['MACD_12_26_9']
                df['Signal Line'] = macd['MACDs_12_26_9']
                df['MACD Buy'] = (df['MACD Line'] > df['Signal Line']) & (df['MACD Line'].shift(1) <= df['Signal Line'].shift(1))
                df['MACD Sell'] = (df['MACD Line'] < df['Signal Line']) & (df['MACD Line'].shift(1) >= df['Signal Line'].shift(1))

        if "Supertrend" in strategies:
            supertrend = df.ta.supertrend(length=7, multiplier=3, append=True)
            if 'SUPERTd_7_3.0' in supertrend.columns:
                df['Supertrend'] = supertrend['SUPERTd_7_3.0']
                df['Supertrend Buy'] = supertrend['SUPERTd_7_3.0'] == 1  # Buy when supertrend is positive
                df['Supertrend Sell'] = supertrend['SUPERTd_7_3.0'] == -1  # Sell when supertrend is negative

        if "Stochastic" in strategies:
            stochastic = df.ta.stoch(append=True)
            if 'STOCHk_14_3_3' in stochastic.columns and 'STOCHd_14_3_3' in stochastic.columns:
                df['Stochastic K'] = stochastic['STOCHk_14_3_3']
                df['Stochastic D'] = stochastic['STOCHd_14_3_3']
                df['Stochastic Buy'] = (df['Stochastic K'] > df['Stochastic D']) & (df['Stochastic K'].shift(1) <= df['Stochastic D'].shift(1))
                df['Stochastic Sell'] = (df['Stochastic K'] < df['Stochastic D']) & (df['Stochastic K'].shift(1) >= df['Stochastic D'].shift(1))

        if "RSI" in strategies:
            df['RSI'] = ta.rsi(df['close'], length=14)
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

        # Filter crashes to keep only one per week (on Fridays)
        df['Crash'] = df['Crash'] & (df.index.weekday == 4)

        # Adjust buy and sell signals based on crashes
        df['Adjusted Sell'] = ((df.get('MACD Sell', False) | df.get('Supertrend Sell', False) | df.get('Stochastic Sell', False) | df.get('RSI Sell', False)) &
                                (~df['Crash'].shift(1).fillna(False)))
        df['Adjusted Buy'] = ((df.get('MACD Buy', False) | df.get('Supertrend Buy', False) | df.get('Stochastic Buy', False) | df.get('RSI Buy', False)) &
                               (~df['Crash'].shift(1).fillna(False)))
    except KeyError as e:
        st.error(f"KeyError: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    return df

# Function to apply T+ holding constraint
def apply_t_plus(df, t_plus):
    t_plus_days = int(t_plus)

    if t_plus_days > 0:
        df['Buy Date'] = np.nan
        df.loc[df['Adjusted Buy'], 'Buy Date'] = df.index[df['Adjusted Buy']]
        df['Buy Date'] = df['Buy Date'].ffill()
        df['Earliest Sell Date'] = df['Buy Date'] + pd.to_timedelta(t_plus_days, unit='D')
        df['Adjusted Sell'] = df['Adjusted Sell'] & (df.index > df['Earliest Sell Date'])

    return df

# Function to run backtesting using vectorbt's from_signals
def run_backtest(df, init_cash, fees, direction, t_plus):
    df = apply_t_plus(df, t_plus)
    entries = df['Adjusted Buy']
    exits = df['Adjusted Sell']

    if entries.empty or exits.empty or not entries.any() or not exits.any():
        return None

    portfolio = vbt.Portfolio.from_signals(
        df['close'],
        entries,
        exits,
        init_cash=init_cash,
        fees=fees,
        direction=direction
    )
    return portfolio

# Calculate crash likelihood
def calculate_crash_likelihood(df):
    crash_counts = df['Crash'].resample('W').sum()
    total_weeks = len(crash_counts)
    crash_weeks = crash_counts[crash_counts > 0].count()
    return crash_weeks / total_weeks if total_weeks > 0 else 0

# Streamlit App
st.title('Mô hình cảnh báo sớm cho các chỉ số và cổ phiếu')
st.write('Ứng dụng này phân tích các cổ phiếu với các tín hiệu mua/bán và cảnh báo sớm trước khi có sự sụt giảm giá mạnh của thị trường chứng khoán trên sàn HOSE và chỉ số VNINDEX.')

# Sidebar for Portfolio Selection
with st.sidebar.expander("Danh mục đầu tư", expanded=True):
    vn30 = VN30()
    selected_stocks = []
    portfolio_options = st.multiselect('Chọn danh mục', ['VN30', 'Chọn mã theo ngành'], key='portfolio_options_main')

    display_vn30 = 'VN30' in portfolio_options  # Set to True only if VN30 is selected

    if 'VN30' in portfolio_options:
        selected_symbols = st.multiselect('Chọn mã cổ phiếu trong VN30', vn30.symbols, default=vn30.symbols, key='vn30_symbols_main')
        
    if 'Chọn mã theo ngành' in portfolio_options:
        selected_sector = st.selectbox('Chọn ngành để lấy dữ liệu', list(SECTOR_FILES.keys()), key='sector_selection_main')
        if selected_sector:
            df_full = load_data(SECTOR_FILES[selected_sector])
            available_symbols = df_full['StockSymbol'].unique().tolist()
            sector_selected_symbols = st.multiselect('Chọn mã cổ phiếu trong ngành', available_symbols, key='sector_symbols_main')
            selected_stocks.extend(sector_selected_symbols)
            display_vn30 = False  # Disable VN30 display if sector is selected

    # Display color key for crash risk
    st.markdown("""
    <div style='margin-top: 20px;'>
        <strong>Chỉ số Đánh Giá Rủi Ro Sụp Đổ:</strong>
        <ul>
            <li><span style='color: #FF5733;'>Màu Đỏ: Rủi Ro Cao</span></li>
            <li><span style='color: #FFC107;'>Màu Vàng: Rủi Ro Trung Bình</span></li>
            <li><span style='color: #4CAF50;'>Màu Xanh Lá: Rủi Ro Thấp</span></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Portfolio tab
with st.sidebar.expander("Thông số kiểm tra", expanded=True):
    init_cash = st.number_input('Vốn đầu tư (VNĐ):', min_value=100_000_000, max_value=1_000_000_000, value=100_000_000, step=1_000_000, key='init_cash_main')
    fees = st.number_input('Phí giao dịch (%):', min_value=0.0, max_value=10.0, value=0.1, step=0.01, key='fees_main') / 100
    direction_vi = st.selectbox("Vị thế", ["Mua", "Bán"], index=0, key='direction_vi_main')
    direction = "longonly" if direction_vi == "Mua" else "shortonly"
    t_plus = st.selectbox("Thời gian nắm giữ tối thiểu", [0, 1, 2.5, 3], index=0, key='t_plus_main')

    # New trading parameters
    take_profit_percentage = st.number_input('Take Profit (%)', min_value=0.0, max_value=100.0, value=10.0, step=0.1, key='take_profit_main')
    stop_loss_percentage = st.number_input('Stop Loss (%)', min_value=0.0, max_value=100.0, value=5.0, step=0.1, key='stop_loss_main')
    trailing_take_profit_percentage = st.number_input('Trailing Take Profit (%)', min_value=0.0, max_value=100.0, value=2.0, step=0.1, key='trailing_take_profit_main')
    trailing_stop_loss_percentage = st.number_input('Trailing Stop Loss (%)', min_value=0.0, max_value=100.0, value=1.5, step=0.1, key='trailing_stop_loss_main')

    # Sidebar: Choose the strategies to apply
    strategies = st.multiselect("Các chỉ báo", ["MACD", "Supertrend", "Stochastic", "RSI"], default=["MACD", "Supertrend", "Stochastic", "RSI"], key='strategies_main')

# Ensure that the date range is within the available data
if selected_stocks:
    if 'VN30' in portfolio_options and 'Chọn mã theo ngành' in portfolio_options:
        sector_data = load_detailed_data(selected_stocks)
        combined_data = pd.concat([vn30_stocks, sector_data])
    elif 'VN30' in portfolio_options:
        combined_data = vn30_stocks
    elif 'Chọn mã theo ngành' in portfolio_options:
        combined_data = load_detailed_data(selected_stocks)
    else:
        combined_data = pd.DataFrame()

    if not combined_data.empty:
        combined_data = combined_data[~combined_data.index.duplicated(keep='first')]  # Ensure unique indices

        # Assuming the combined data already covers today's data for VN30 and possibly other dates for sector stocks
        first_available_date = combined_data.index.min().date()
        last_available_date = combined_data.index.max().date()

        # Ensure selected date range is within the available data range
        start_date = st.date_input('Ngày bắt đầu', first_available_date, key='start_date_main')
        end_date = st.date_input('Ngày kết thúc', last_available_date, key='end_date_main')

        if start_date < first_available_date:
            start_date = first_available_date
            st.warning("Ngày bắt đầu đã được điều chỉnh để nằm trong phạm vi dữ liệu có sẵn.")

        if end_date > last_available_date:
            end_date = last_available_date
            st.warning("Ngày kết thúc đã được điều chỉnh để nằm trong phạm vi dữ liệu có sẵn.")

        if start_date >= end_date:
            st.error("Lỗi: Ngày kết thúc phải sau ngày bắt đầu.")
        else:
            try:
                df_filtered = ensure_datetime_compatibility(start_date, end_date, combined_data)

                if df_filtered.empty:
                    st.error("Không có dữ liệu cho khoảng thời gian đã chọn.")
                else:
                    # Calculate indicators and crashes
                    df_filtered = calculate_indicators_and_crashes(df_filtered, strategies)

                    # Run backtest
                    portfolio = run_backtest(df_filtered, init_cash, fees, direction, t_plus)

                    if portfolio is None or len(portfolio.orders.records) == 0:
                        st.error("Không có giao dịch nào được thực hiện trong khoảng thời gian này.")
                    else:
                        # Create tabs for different views on the main screen
                        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Tóm tắt", "Chi tiết kết quả kiểm thử", "Tổng hợp lệnh mua/bán", "Đường cong giá trị", "Biểu đồ", "Danh mục đầu tư"])
                        
                        with tab1:
                            try:
                                st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Tóm tắt chiến lược</h2>", unsafe_allow_html=True)
                                
                                # Hiển thị tên chỉ báo và tỷ lệ thắng
                                indicator_name = ", ".join(strategies)
                                win_rate = portfolio.stats()['Win Rate [%]']
                                win_rate_color = "#4CAF50" if win_rate > 50 else "#FF5733"
                        
                                st.markdown(f"<div style='text-align: center; margin-bottom: 20px;'><span style='color: {win_rate_color}; font-size: 24px; font-weight: bold;'>Tỷ lệ thắng: {win_rate:.2f}%</span><br><span style='font-size: 18px;'>Sử dụng chỉ báo: {indicator_name}</span></div>", unsafe_allow_html=True)
                        
                                # Mục hiệu suất
                                cumulative_return = portfolio.stats()['Total Return [%]']
                                annualized_return = portfolio.stats().get('Annual Return [%]', 0)
                                st.markdown("<div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin-bottom: 20px;'>", unsafe_allow_html=True)
                                st.markdown(f"<p style='text-align: center; margin: 0;'><strong>Hiệu suất trên các mã chọn: {', '.join(selected_stocks)}</strong></p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='text-align: center; margin: 0;'><strong>Tổng lợi nhuận: {cumulative_return:.2f}%</strong> | <strong>Lợi nhuận hàng năm: {annualized_return:.2f}%</strong></p>", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                        
                                # Đồ thị giá và điểm sụt giảm
                                price_data = df_filtered['close']
                                crash_df = df_filtered[df_filtered['Crash']]
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=price_data.index, y=price_data, mode='lines', name='Giá', line=dict(color='#1f77b4')))
                                fig.add_trace(go.Scatter(x=crash_df.index, y=crash_df['close'], mode='markers', marker=dict(color='orange', size=8, symbol='triangle-down'), name='Điểm sụt giảm'))
                        
                                fig.update_layout(
                                    title="Biểu đồ Giá cùng Điểm Sụt Giảm",
                                    xaxis_title="Ngày",
                                    yaxis_title="Giá",
                                    legend_title="Chú thích",
                                    template="plotly_white"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                                # Xem chi tiết các điểm sụt giảm
                                crash_details = crash_df[['close']]
                                crash_details.reset_index(inplace=True)
                                crash_details.rename(columns={'Datetime': 'Ngày Sụt Giảm', 'close': 'Giá'}, inplace=True)
                                
                                if st.button('Xem Chi Tiết', key='detail_button'):
                                    st.markdown("**Danh sách các điểm sụt giảm:**")
                                    st.dataframe(crash_details.style.format(subset=['Giá'], formatter="{:.2f}"), height=300)
                        
                            except Exception as e:
                                st.error(f"Đã xảy ra lỗi: {e}")
                        

                        with tab2:
                            st.markdown("**Chi tiết kết quả kiểm thử:**")
                            st.markdown("Tab này hiển thị hiệu suất tổng thể của chiến lược giao dịch đã chọn. \
                                        Bạn sẽ tìm thấy các chỉ số quan trọng như tổng lợi nhuận, lợi nhuận/lỗ, và các thống kê liên quan khác.")
                            stats_df = pd.DataFrame(portfolio.stats(), columns=['Giá trị'])
                            stats_df.index.name = 'Chỉ số'
                            metrics_vi = {
                                'Start Value': 'Giá trị ban đầu',
                                'End Value': 'Giá trị cuối cùng',
                                'Total Return [%]': 'Tổng lợi nhuận [%]',
                                'Max Drawdown [%]': 'Mức giảm tối đa [%]',
                                'Total Trades': 'Tổng số giao dịch',
                                'Win Rate [%]': 'Tỷ lệ thắng [%]',
                                'Best Trade [%]': 'Giao dịch tốt nhất [%]',
                                'Worst Trade [%]': 'Giao dịch tệ nhất [%]',
                                'Profit Factor': 'Hệ số lợi nhuận',
                                'Expectancy': 'Kỳ vọng',
                                'Sharpe Ratio': 'Tỷ lệ Sharpe',
                                'Sortino Ratio': 'Tỷ lệ Sortino',
                                'Calmar Ratio': 'Tỷ lệ Calmar'
                            }
                            stats_df.rename(index=metrics_vi, inplace=True)
                            st.dataframe(stats_df, height=800)

                        with tab3:
                            st.markdown("**Tổng hợp lệnh mua/bán:**")
                            st.markdown("Tab này cung cấp danh sách chi tiết của tất cả các lệnh mua/bán được thực hiện bởi chiến lược. \
                                        Bạn có thể phân tích các điểm vào và ra của từng giao dịch, cùng với lợi nhuận hoặc lỗ.")
                            trades_df = portfolio.trades.records_readable
                            trades_df = trades_df.round(2)
                            trades_df.index.name = 'Số giao dịch'
                            trades_df.drop(trades_df.columns[[0, 1]], axis=1, inplace=True)
                            st.dataframe(trades_df, width=800, height=600)

                        equity_data = portfolio.value()
                        drawdown_data = portfolio.drawdown() * 100

                        with tab4:
                            equity_trace = go.Scatter(x=equity_data.index, y=equity_data, mode='lines', name='Giá trị', line=dict(color='green'))
                            equity_fig = go.Figure(data=[equity_trace])
                            equity_fig.update_layout(
                                title='Đường cong giá trị',
                                xaxis_title='Ngày',
                                yaxis_title='Giá trị',
                                width=800,
                                height=600
                            )
                            st.plotly_chart(equity_fig)
                            st.markdown("**Đường cong giá trị:**")
                            st.markdown("Biểu đồ này hiển thị sự tăng trưởng giá trị danh mục của bạn theo thời gian, \
                                        cho phép bạn thấy cách chiến lược hoạt động trong các điều kiện thị trường khác nhau.")

                        with tab5:
                            fig = portfolio.plot()
                            crash_df = df_filtered[df_filtered['Crash']]
                            fig.add_scatter(
                                x=crash_df.index,
                                y=crash_df['close'],
                                mode='markers',
                                marker=dict(color='orange', size=10, symbol='triangle-down'),
                                name='Sụt giảm'
                            )
                            st.markdown("**Biểu đồ:**")
                            st.markdown("Biểu đồ tổng hợp này kết hợp đường cong giá trị với các tín hiệu mua/bán và cảnh báo sụp đổ tiềm năng, \
                                        cung cấp cái nhìn tổng thể về hiệu suất của chiến lược.")
                            st.plotly_chart(fig, use_container_width=True)

                        with tab6:
                            st.markdown("**Danh mục đầu tư:**")
                            st.markdown("Danh sách các mã cổ phiếu theo danh mục.")
                            optimizer = PortfolioOptimizer()
                            df_selected_stocks = df_filtered[df_filtered['StockSymbol'].isin(selected_stocks)]
                            data_matrix = df_selected_stocks.pivot_table(values='close', index=df_selected_stocks.index, columns='StockSymbol').dropna()
                            optimal_weights = optimizer.MSR_portfolio(data_matrix.values)

                            st.write("Optimal Weights for Selected Stocks:")
                            for stock, weight in zip(data_matrix.columns, optimal_weights):
                                st.write(f"{stock}: {weight:.4f}")

                        crash_likelihoods = {}
                        for stock in selected_stocks:
                            stock_df = df_filtered[df_filtered['StockSymbol'] == stock]
                            crash_likelihoods[stock] = calculate_crash_likelihood(stock_df)

                        if crash_likelihoods:
                            st.markdown("**Xác suất sụt giảm:**")
                            crash_likelihoods_df = pd.DataFrame(list(crash_likelihoods.items()), columns=['Stock', 'Crash Likelihood'])
                            crash_likelihoods_df.set_index('Stock', inplace=True)
                            fig, ax = plt.subplots(figsize=(10, len(crash_likelihoods_df) / 2))
                            sns.heatmap(crash_likelihoods_df, annot=True, cmap='RdYlGn_r', ax=ax)
                            st.pyplot(fig)
            except KeyError as e:
                st.error(f"Key error: {e}")
            except Exception as e:
                if 'tuple index out of range' not in str(e):
                    st.error(f"An unexpected error occurred: {e}")

else:
    st.write("Please select a portfolio or sector to view data.")
