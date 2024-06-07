import streamlit as st
import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize
from scipy.signal import find_peaks
import plotly.graph_objects as go
import seaborn as sns
import vectorbt as vbt
import pandas_ta as ta

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

PORTFOLIO_FILES = {
    'VN30': 'VN30.csv',
    'VN100': 'VN100.csv',
    'VNAllShare': 'VNAllShare.csv'
}

@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()
    return pd.read_csv(file_path, parse_dates=['Datetime'], dayfirst=True).set_index('Datetime')

def load_portfolio_symbols(portfolio_name):
    file_path = PORTFOLIO_FILES.get(portfolio_name, '')
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return []
    return pd.read_csv(file_path)['symbol'].tolist()

# Load and filter detailed data
def load_detailed_data(selected_stocks):
    data = pd.DataFrame()
    for sector, file_path in SECTOR_FILES.items():
        df = load_data(file_path)
        if not df.empty:
            sector_data = df[df['StockSymbol'].isin(selected_stocks)]
            data = pd.concat([data, sector_data])
    return data

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
    portfolio_options = st.multiselect('Chọn danh mục', ['VN30', 'VN100', 'VNAllShare'])
    selected_stocks = []
    selected_sector = None

    if portfolio_options:
        for portfolio_option in portfolio_options:
            symbols = load_portfolio_symbols(portfolio_option)
            if symbols:
                selected_symbols = st.multiselect(f'Chọn mã cổ phiếu trong {portfolio_option}', symbols, default=symbols)
                selected_stocks.extend(selected_symbols)

    selected_sector = st.selectbox('Chọn ngành để lấy dữ liệu:', list(SECTOR_FILES.keys()))
    if selected_sector:
        df_full = load_data(SECTOR_FILES[selected_sector])
        if not selected_stocks:
            available_symbols = df_full['StockSymbol'].unique().tolist()
            selected_stocks = st.multiselect('Chọn mã cổ phiếu trong ngành', available_symbols)

# Portfolio tab
with st.sidebar.expander("Thông số kiểm tra", expanded=True):
    init_cash = st.number_input('Vốn đầu tư (VNĐ):', min_value=100_000_000, max_value=1_000_000_000, value=100_000_000, step=1_000_000)
    fees = st.number_input('Phí giao dịch (%):', min_value=0.0, max_value=10.0, value=0.1, step=0.01) / 100
    direction_vi = st.selectbox("Vị thế", ["Mua", "Bán"], index=0)
    direction = "longonly" if direction_vi == "Mua" else "shortonly"
    t_plus = st.selectbox("Thời gian nắm giữ tối thiểu", [0, 1, 2.5, 3], index=0)

    # New trading parameters
    take_profit_percentage = st.number_input('Take Profit (%)', min_value=0.0, max_value=100.0, value=10.0, step=0.1)
    stop_loss_percentage = st.number_input('Stop Loss (%)', min_value=0.0, max_value=100.0, value=5.0, step=0.1)
    trailing_take_profit_percentage = st.number_input('Trailing Take Profit (%)', min_value=0.0, max_value=100.0, value=2.0, step=0.1)
    trailing_stop_loss_percentage = st.number_input('Trailing Stop Loss (%)', min_value=0.0, max_value=100.0, value=1.5, step=0.1)

    # Sidebar: Choose the strategies to apply
    strategies = st.multiselect("Các chỉ báo", ["MACD", "Supertrend", "Stochastic", "RSI"], default=["MACD", "Supertrend", "Stochastic", "RSI"])

# Ensure that the date range is within the available data
if selected_stocks:
    df_full = load_detailed_data(selected_stocks)
    
    if not df_full.empty:
        first_available_date = df_full.index.min().date()
        last_available_date = df_full.index.max().date()

        # Ensure selected date range is within the available data range
        start_date = st.date_input('Ngày bắt đầu', first_available_date)
        end_date = st.date_input('Ngày kết thúc', last_available_date)

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
                df_filtered = df_full.loc[start_date:end_date]

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
                        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Tóm tắt", "Chi tiết kết quả kiểm thử", "Tổng hợp lệnh mua/bán", "Đường cong giá trị", "Mức sụt giảm tối đa", "Biểu đồ", "Danh mục đầu tư"])

                        with tab1:
                            st.markdown("**Tóm tắt:**")
                            st.markdown("Tab này hiển thị các chỉ số quan trọng như tổng lợi nhuận, tỷ lệ thắng, và mức sụt giảm tối đa.")
                            summary_stats = portfolio.stats()[['Total Return [%]', 'Win Rate [%]', 'Max Drawdown [%]']]
                            metrics_vi_summary = {
                                'Total Return [%]': 'Tổng lợi nhuận [%]',
                                'Win Rate [%]': 'Tỷ lệ thắng [%]',
                                'Max Drawdown [%]': 'Mức sụt giảm tối đa [%]'
                            }
                            summary_stats.rename(index=metrics_vi_summary, inplace=True)

                            for index, value in summary_stats.items():
                                st.markdown(f'<div class="highlight">{index}: {value}</div>', unsafe_allow_html=True)

                            # Add crash details
                            crash_details = df_filtered[df_filtered['Crash']][['close']]
                            crash_details.reset_index(inplace=True)
                            crash_details.rename(columns={'Datetime': 'Ngày crash', 'close': 'Giá'}, inplace=True)
                            st.markdown("**Danh sách các điểm crash:**")
                            st.dataframe(crash_details, height=200)

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
                            drawdown_trace = go.Scatter(
                                x=drawdown_data.index,
                                y=drawdown_data,
                                mode='lines',
                                name='Mức sụt giảm tối đa',
                                fill='tozeroy',
                                line=dict(color='red')
                            )
                            drawdown_fig = go.Figure(data=[drawdown_trace])
                            drawdown_fig.update_layout(
                                title='Mức sụt giảm tối đa',
                                xaxis_title='Ngày',
                                yaxis_title='% Mức sụt giảm tối đa',
                                template='plotly_white',
                                width=800,
                                height=600
                            )
                            st.plotly_chart(drawdown_fig)
                            st.markdown("**Mức sụt giảm tối đa:**")
                            st.markdown("Biểu đồ này minh họa sự sụt giảm từ đỉnh đến đáy của danh mục của bạn, \
                                        giúp bạn hiểu rõ hơn về tiềm năng thua lỗ của chiến lược.")

                        with tab6:
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

                        with tab7:
                            st.markdown("**Danh mục đầu tư:**")
                            st.markdown("Danh sách các mã cổ phiếu theo danh mục VN100, VN30 và VNAllShare.")
                            optimizer = PortfolioOptimizer()
                            df_selected_stocks = df_filtered[df_filtered['StockSymbol'].isin(selected_stocks)]
                            data_matrix = df_selected_stocks.pivot_table(values='close', index=df_selected_stocks.index, columns='StockSymbol').dropna()
                            optimal_weights = optimizer.MSR_portfolio(data_matrix.values)

                            st.write("Optimal Weights for Selected Stocks:")
                            for stock, weight in zip(data_matrix.columns, optimal_weights):
                                st.write(f"{stock}: {weight:.4f}")

                            for portfolio_option in portfolio_options:
                                symbols = load_portfolio_symbols(portfolio_option)
                                st.markdown(f"**{portfolio_option}:**")
                                st.write(symbols)

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
                st.error(f"An unexpected error occurred: {e}")

else:
    st.write("Please select a portfolio or sector to view data.")
