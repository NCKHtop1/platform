import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
from scipy.optimize import minimize
from scipy.signal import find_peaks
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import vectorbt as vbt
import pandas_ta as ta

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

@st.cache(allow_output_mutation=True)
def load_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()
    df = pd.read_csv(file_path, parse_dates=['Datetime'], dayfirst=True)
    df.set_index('Datetime', inplace=True)
    return df

def load_portfolio_symbols(portfolio_name):
    file_path = PORTFOLIO_FILES.get(portfolio_name, '')
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return []
    return pd.read_csv(file_path)['symbol'].tolist()

# Ensure datetime comparison compatibility
def ensure_datetime_compatibility(start_date, end_date, df):
    if not isinstance(start_date, pd.Timestamp):
        start_date = pd.Timestamp(start_date)
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.Timestamp(end_date)
    return df[(df.index >= start_date) & (df.index <= end_date)]

# Load and filter detailed data
def load_detailed_data(selected_stocks):
    data = pd.DataFrame()
    for sector, file_path in SECTOR_FILES.items():
        df = load_data(file_path)
        if not df.empty:
            sector_data = df[df['StockSymbol'].isin(selected_stocks)]
            data = pd.concat([data, sector_data])
    return data

class PortfolioOptimizer:
    def MSR_portfolio(self, data: np.ndarray) -> np.ndarray:
        """
        Markowitz Maximum Sharpe-Ratio Portfolio (MSRP)
        Returns the optimal asset weights for the MSRP portfolio, given historical price data of assets.

        Original equation to solve:
            max (w^T mu) / sqrt(w^T Sigma w)
            subject to sum(w) = 1

        Quadratic Programming (QP) reformulation:
            min (1/2) * w^T Sigma w
            subject to w^T mu = 1
                      sum(w) = 1

        Args:
            data (np.ndarray): Historical price data of assets

        Returns:
            w (np.ndarray): Optimal asset weights
        """
        X = np.diff(np.log(data), axis=0)           # Calculate log returns from historical price data
        mu = np.mean(X, axis=0)                     # Calculate the mean returns of the assets
        Sigma = np.cov(X, rowvar=False)             # Calculate the covariance matrix of the returns

        w = self.MSRP_solver(mu, Sigma)             # Use the MSRP solver to get the optimal weights
        return w                                    # Return the optimal weights

    def MSRP_solver(self, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        """
        Method for solving Markowitz Maximum Sharpe-Ratio Portfolio (MSRP)
        Returns the optimal asset weights for the MSRP portfolio, given mean returns and covariance matrix.

        Original equation to solve:
            max (w^T mu) / sqrt(w^T Sigma w)
            subject to sum(w) = 1

        Quadratic Programming (QP) reformulation:
            min (1/2) * w^T Sigma w
            subject to w^T mu = 1
                      sum(w) = 1

        Args:
            mu (np.ndarray): Mean returns
            Sigma (np.ndarray): Covariance matrix

        Returns:
            w (np.ndarray): Optimal asset weights
        """
        N = Sigma.shape[0]          # Number of assets (stocks)
        if np.all(mu <= 1e-8):      # Check if mean returns are close to zero
            return np.zeros(N)      # Return zero weights if no returns

        Dmat = 2 * Sigma                    # Quadratic term for the optimizer
        Amat = np.vstack((mu, np.ones(N)))  # Combine mean returns and sum constraint for constraints
        bvec = np.array([1, 1])             # Right-hand side of constraints (1 for mean returns and sum)
        dvec = np.zeros(N)                  # Linear term (zero for this problem)

        # Call the QP solver
        w = self.solve_QP(Dmat, dvec, Amat, bvec, meq=2)
        return w / np.sum(abs(w))                # Normalize weights to sum to 1

    def solve_QP(self, Dmat: np.ndarray, dvec: np.ndarray, Amat: np.ndarray, bvec: np.ndarray, meq: int = 0) -> np.ndarray:
        """
        Quadratic programming solver.
        Returns the optimal asset weights for the QP problem.

        Args:
            Dmat (np.ndarray): Matrix of quadratic coefficients
            dvec (np.ndarray): Vector of linear coefficients
            Amat (np.ndarray): Matrix of linear constraints
            bvec (np.ndarray): Vector of linear constraints
            meq (int): Number of equality constraints

        Returns:
            x (np.ndarray): Optimal asset weights
        """
        def portfolio_obj(x):
            """
            Objective function for the QP problem.
            Minimize 0.5 * x^T * Dmat * x + dvec^T * x.
            """
            return 0.5 * np.dot(x, np.dot(Dmat, x)) + np.dot(dvec, x)

        def portfolio_constr_eq(x):
            """
            Equality constraints for the QP problem.
            """
            return np.dot(Amat[:meq], x) - bvec[:meq]

        def portfolio_constr_ineq(x):
            """
            Inequality constraints for the QP problem.
            """
            if Amat.shape[0] - meq == 0:
                return np.array([])
            else:
                return np.dot(Amat[meq:], x) - bvec[meq:]

        # Define constraints for the optimizer
        cons = [{'type': 'eq', 'fun': portfolio_constr_eq}]

        if meq < len(bvec):
            cons.append({'type': 'ineq', 'fun': portfolio_constr_ineq})

        # Initial guess for the weights
        initial_guess = np.ones(Dmat.shape[0]) / Dmat.shape[0]

        # Use the 'SLSQP' method to minimize the objective function subject to constraints
        res = minimize(portfolio_obj, initial_guess, constraints=cons, method='SLSQP')

        # Check if the optimization was successful
        if not res.success:
            raise ValueError('Quadratic programming failed to find a solution.')

        # Return the optimal weights
        return res.x

    def GMV_portfolio(self, data: np.ndarray, shrinkage: bool =False, shrinkage_type = 'ledoit', shortselling: bool =True, leverage: int =None) -> np.ndarray:
        """
        Global Minimum Variance Portfolio
        Returns the optimal asset weights for the GMVP, given historical price data of assets.

        Args:
            data (np.ndarray): Historical price data of assets
            shrinkage (bool): Flag to use Ledoit-Wolf shrinkage estimator
            shortselling (bool): Flag to allow short-selling
            leverage (int): Leverage factor

        Returns:
            w (np.ndarray): Optimal asset weights
        """
        X = np.diff(np.log(data), axis=0)
        X = X[~np.isnan(X).any(axis=1)]  # Remove rows with NaN values

        # Calculate covariance matrix
        if shrinkage:
            if shrinkage_type == 'ledoit':
                # Use Ledoit-Wolf shrinkage estimator
                Sigma = self.ledoit_wolf_shrinkage(X)
            elif shrinkage_type == 'ledoit_cc':
                # Use Ledoit-Wolf shrinkage estimator with custom covariance
                Sigma = self.ledoitwolf_cc(X)
            elif shrinkage_type == 'oas':
                # Use Oracle Approximating Shrinkage (OAS) estimator
                Sigma = self.oas_shrinkage(X)
            elif shrinkage_type == 'graphical_lasso':
                # Use Graphical Lasso estimator
                Sigma = self.graphical_lasso_shrinkage(X)
            elif shrinkage_type == 'mcd':
                # Use Minimum Covariance Determinant (MCD) estimator
                Sigma = self.mcd_shrinkage(X)
            else:
                raise ValueError('Invalid shrinkage type. Choose from: ledoit, ledoit_cc, oas, graphical_lasso, mcd')
        else:
            Sigma = np.cov(X, rowvar=False)

        if not shortselling:
            N = Sigma.shape[0]
            Dmat = 2 * Sigma
            Amat = np.vstack((np.ones(N), np.eye(N)))
            bvec = np.array([1] + [0] * (N+1))  # Update bvec to have length N+1
            dvec = np.zeros(N)
            w = self.solve_QP(Dmat, dvec, Amat, bvec, meq=1)
        else:
            ones = np.ones(Sigma.shape[0])
            w = np.linalg.solve(Sigma, ones)
            w /= np.sum(w)

        if leverage is not None and leverage < np.inf:
            w = leverage * w / np.sum(np.abs(w))

        return w

    def ledoitwolf_cc(self, returns: np.ndarray) -> np.ndarray:
        """
        Implementation of Ledoit-Wolf shrinkage estimator with custom covariance.
        The Ledoit-Wolf shrinkage method is a technique used to improve the estimation of
        the covariance matrix by shrinking the sample covariance matrix towards a target matrix.

        Returns the Ledoit-Wolf shrinkage covariance matrix.

        Args:
            returns (np.ndarray): Historical returns data

        Returns:
            np.ndarray: Ledoit-Wolf shrinkage covariance matrix
        """
        T, N = returns.shape
        returns = returns - np.mean(returns, axis=0, keepdims=True)  # Subtract mean using numpy mean
        df = pd.DataFrame(returns)
        Sigma = df.cov().values
        Cor = df.corr().values
        diagonals = np.diag(Sigma)
        var = diagonals.reshape(len(Sigma), 1)
        vols = var ** 0.5

        rbar = np.mean((Cor.sum(1) - 1) / (Cor.shape[1] - 1))
        cc_cor = np.matrix([[rbar] * N for _ in range(N)])
        np.fill_diagonal(cc_cor, 1)
        F = np.diag((diagonals ** 0.5)) @ cc_cor @ np.diag((diagonals ** 0.5))  # vol-cor decomposition

        y = returns ** 2
        mat1 = (y.transpose() @ y) / T - Sigma ** 2  # y is centered, cross term cancels
        pihat = mat1.sum()

        mat2 = ((returns ** 3).transpose() @ returns) / T - var * Sigma  # cross term cancels
        np.fill_diagonal(mat2, 0)
        rhohat = np.diag(mat1).sum() + rbar * ((1 / vols) @ vols.transpose() * mat2).sum()
        gammahat = np.linalg.norm(Sigma - F, "fro") ** 2
        kappahat = (pihat - rhohat) / gammahat
        delta = max(0, min(1, kappahat / T))

        return delta * F + (1 - delta) * Sigma

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
                
                if selected_symbols:
                    optimizer = PortfolioOptimizer()
                    df_selected_stocks = load_detailed_data(selected_symbols)
                    data_matrix = df_selected_stocks.pivot_table(values='close', index=df_selected_stocks.index, columns='StockSymbol').dropna()
                    optimal_weights = optimizer.MSR_portfolio(data_matrix.values)

                    st.write("Optimal Weights for Selected Stocks:")
                    for stock, weight in zip(data_matrix.columns, optimal_weights):
                        st.write(f"{stock}: {weight:.4f}")

                    st.stop()  # Stop execution to prevent displaying the backtesting screen

    else:
        selected_sector = st.selectbox('Chọn ngành', list(SECTOR_FILES.keys()))
        df_full = load_data(SECTOR_FILES[selected_sector])
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
        try:
            # Convert dates only once and use converted dates for comparisons
            first_available_date = pd.Timestamp(df_full.index.min())
            last_available_date = pd.Timestamp(df_full.index.max())

            start_date = st.date_input('Ngày bắt đầu', first_available_date)
            end_date = st.date_input('Ngày kết thúc', last_available_date)

            # Convert user input dates to timestamps if not already
            start_date = pd.Timestamp(start_date) if not isinstance(start_date, pd.Timestamp) else start_date
            end_date = pd.Timestamp(end_date) if not isinstance(end_date, pd.Timestamp) else end_date

            # Ensure date range is within limits and use the ensured function
            if start_date < first_available_date:
                start_date = first_available_date
                st.warning("Ngày bắt đầu đã được điều chỉnh để nằm trong phạm vi dữ liệu có sẵn.")
            if end_date > last_available_date:
                end_date = last_available_date
                st.warning("Ngày kết thúc đã được điều chỉnh để nằm trong phạm vi dữ liệu có sẵn.")

            if start_date >= end_date:
                st.error("Lỗi: Ngày kết thúc phải sau ngày bắt đầu.")
            else:
                df_filtered = ensure_datetime_compatibility(start_date, end_date, df_full)

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
                            df_selected_stocks = df_full[df_full['StockSymbol'].isin(selected_stocks)]
                            data_matrix = df_selected_stocks.pivot_table(values='close', index=df_selected_stocks.index, columns='StockSymbol').dropna()
                            optimal_weights = optimizer.MSR_portfolio(data_matrix.values)

                            st.write("Optimal Weights for Selected Stocks:")
                            for stock, weight in zip(data_matrix.columns, optimal_weights):
                                st.write(f"{stock}: {weight:.4f}")

                            for portfolio_option in portfolio_options:
                                symbols = load_portfolio_symbols(portfolio_option)
                                st.markdown(f"**{portfolio_option}:**")
                                st.write(symbols)

                        # Calculate crash likelihood for each selected stock and plot heatmap
                        crash_likelihoods = {}
                        for stock in selected_stocks:
                            stock_df = df_filtered[df_filtered['StockSymbol'] == stock]
                            crash_likelihoods[stock] = calculate_crash_likelihood(stock_df)

                        # Plot heatmap
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
