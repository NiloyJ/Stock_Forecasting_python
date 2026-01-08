# import streamlit as st
# from pages.utils.modal_train import (
#     get_data,
#     get_rolling_mean,
#     get_differencing_order,
#     stationary_check,
#     fit_model,
#     evaluate_model,
#     scaling,
#     inverse_scaling,
#     get_forecast
# )
# import pandas as pd 
# import yfinance as yf
# import plotly.graph_objects as go
# import datetime 

# st.set_page_config(
#     page_title="Stock Prediction Page",
#     page_icon=":bar_chart:",
#     layout="wide"
# )

# st.title("Stock Prediction Page :bar_chart:")

# col1, col2, col3 = st.columns(3)

# today = datetime.date.today()

# with col1:
#     ticker = st.text_input("Enter Stock Ticker", value="AAPL")
# with col2:
#     start_date = st.date_input("Choose start date", datetime.date(today.year - 1, today.month, today.day))  

# with col3: 
#     end_date = st.date_input("Choose start date", datetime.date(today.year, today.month, today.day))  

# st.subheader(ticker)

# stock = yf.Ticker(ticker)

# st.write(stock.info['longBusinessSummary'])
# st.write(stock.info['sector'])
# st.write(stock.info['industry'])
# st.write("Market Cap:", stock.info['marketCap'])
# st.write(stock.info['website'])

# col1, col2 = st.columns(2)

# with col1:
#     df = pd.DataFrame(index = ['Market cap', 'Beta', 'EPS', 'PE ratio'])
#     df[''] = [stock.info['marketCap'], stock.info['beta'],  stock.info['trailingEps'], stock.info['trailingPE']]

# rmse = 0 

# st.subheader('Predicting the next 30 days closing price'+ticker)

# close_price = get_data(ticker)
# rolling_price = get_rolling_mean(close_price)

# differencing_order = get_differencing_order(rolling_price)
# scaled_data, scaler = scaling(close_price)
# rmse = evaluate_model(scaled_data, differencing_order)

# st.write('Model RMSE score', rmse)

# forecast = get_forecast(scaled_data, differencing_order)

# forecast['Close'] = inverse_scaling(scaler, forecast['Close'])

# st.write('Forecast data for next 30 days')

# fig_tail = plotly_table(forecast.sort_index(ascending=True).round(3))
# fig_tail.update_layout(height=300)
# st.plotly_chart(fig_tail, use_container_width=True)

# forecast = pd.concat([rolling_price, forecast])

# st.plotly_chart(Moving_average_forecast(forecast.iloc[150:]), use_container_width= True)

# import streamlit as st
# import pandas as pd
# import yfinance as yf
# import plotly.graph_objects as go
# import numpy as np
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.stattools import adfuller
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error
# from datetime import datetime, timedelta

# # ---------------------------
# # Utility functions
# # ---------------------------
# def get_data(ticker):
#     """Download stock Close price from yfinance"""
#     stock_data = yf.download(ticker, start='2024-01-01')
#     return stock_data[['Close']].dropna()

# def stationary_check(series):
#     """Return p-value from ADF test"""
#     series = series.dropna()
#     adf_test = adfuller(series)
#     return round(adf_test[1], 3)

# def get_rolling_mean(series, window=12):
#     """Return rolling mean series"""
#     return series.rolling(window=window).mean()

# def get_differencing_order(series):
#     """Return differencing order required for stationarity"""
#     series = series.dropna()
#     d = 0
#     p_value = stationary_check(series)
#     while p_value > 0.05:
#         series = series.diff().dropna()
#         p_value = stationary_check(series)
#         d += 1
#     return d

# def fit_model(data, differencing_order, forecast_steps=30):
#     """Fit ARIMA and return forecast"""
#     model = ARIMA(data, order=(5, differencing_order, 0))
#     model_fit = model.fit()
#     forecast = model_fit.get_forecast(steps=forecast_steps)
#     return forecast.predicted_mean

# def evaluate_model(data, differencing_order, test_size=30):
#     """Return RMSE of last test_size predictions"""
#     train, test = data[:-test_size], data[-test_size:]
#     predictions = fit_model(train, differencing_order, forecast_steps=test_size)
#     rmse = np.sqrt(mean_squared_error(test, predictions))
#     return round(rmse, 2)

# def scaling(series):
#     """Scale data and return scaler"""
#     scaler = StandardScaler()
#     scaled = scaler.fit_transform(np.array(series).reshape(-1, 1))
#     return scaled, scaler

# def inverse_scaling(scaler, scaled_data):
#     """Inverse scaling"""
#     return scaler.inverse_transform(np.array(scaled_data).reshape(-1,1))

# def get_forecast(data, differencing_order, forecast_steps=30):
#     """Return forecast DataFrame for next forecast_steps days"""
#     predictions = fit_model(data, differencing_order, forecast_steps=forecast_steps)
#     start_date = datetime.now()
#     dates = pd.date_range(start=start_date, periods=forecast_steps, freq='D')
#     forecast_df = pd.DataFrame(predictions, index=dates, columns=['Forecasted Close Price'])
#     return forecast_df

# def plotly_table(dataframe):
#     """Simple Plotly table"""
#     header_color = 'blue'
#     row_even_color = '#f8fafd'
#     row_odd_color = '#e1efff'

#     fig = go.Figure(
#         data=[go.Table(
#             header=dict(
#                 values=["<b></b>"] + [f"<b>{col}</b>" for col in dataframe.columns],
#                 fill_color=header_color,
#                 line_color=header_color,
#                 align="center",
#                 font=dict(color="white", size=12),
#                 height=35
#             ),
#             cells=dict(
#                 values=[dataframe.index.astype(str).tolist()] + [dataframe[col].tolist() for col in dataframe.columns],
#                 fill_color=[[row_odd_color, row_even_color]*len(dataframe)],
#                 align="left",
#                 font=dict(color="darkslategray", size=12),
#                 height=30
#             )
#         )]
#     )
#     fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0))
#     return fig

# def plot_ohlc(dataframe):
#     """Plot OHLC lines"""
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=dataframe.index, y=dataframe['Open'], mode='lines', name='Open', line=dict(width=2, color='#5ab7ff')))
#     fig.add_trace(go.Scatter(x=dataframe.index, y=dataframe['High'], mode='lines', name='High', line=dict(width=2, color='#00cc96')))
#     fig.add_trace(go.Scatter(x=dataframe.index, y=dataframe['Low'], mode='lines', name='Low', line=dict(width=2, color='#ffa600')))
#     fig.add_trace(go.Scatter(x=dataframe.index, y=dataframe['Close'], mode='lines', name='Close', line=dict(width=2, color='#ff6f61')))
#     fig.update_layout(title='Stock OHLC', xaxis_title='Date', yaxis_title='Price', template='plotly_white', height=500)
#     return fig

# # ---------------------------
# # Streamlit Page
# # ---------------------------
# st.set_page_config(page_title="Stock Prediction", page_icon=":bar_chart:", layout="wide")
# st.title("Stock Prediction Page :bar_chart:")

# col1, col2, col3 = st.columns(3)

# today = datetime.today()

# with col1:
#     ticker = st.text_input("Enter Stock Ticker", value="AAPL")
# with col2:
#     start_date = st.date_input("Choose start date", datetime(today.year-1, today.month, today.day))
# with col3:
#     end_date = st.date_input("Choose end date", today)

# # Fetch stock info
# stock = yf.Ticker(ticker)
# st.subheader(ticker)
# st.write(stock.info.get('longBusinessSummary', 'No description available'))
# st.write("Sector:", stock.info.get('sector', 'N/A'))
# st.write("Industry:", stock.info.get('industry', 'N/A'))
# st.write("Market Cap:", stock.info.get('marketCap', 'N/A'))
# st.write("Website:", stock.info.get('website', 'N/A'))

# # Fetch price data
# close_price = get_data(ticker)
# rolling_price = get_rolling_mean(close_price).dropna()

# # Differencing order
# d_order = get_differencing_order(close_price)

# # Scaling
# scaled_data, scaler = scaling(close_price)

# # RMSE
# rmse = evaluate_model(close_price.values, d_order)
# st.write('Model RMSE score:', rmse)

# # Forecast
# forecast = get_forecast(close_price.values, d_order)
# st.subheader("Forecast for next 30 days")
# st.plotly_chart(plotly_table(forecast.round(2)), use_container_width=True)

# # Merge rolling and forecast for visualization
# combined = pd.concat([rolling_price, forecast.rename(columns={'Forecasted Close Price':'Close'})], axis=1)
# combined = combined.fillna(method='ffill')

# # Plot OHLC (simplified: Open/High/Low = Close for demo)
# combined_ohlc = pd.DataFrame({
#     'Open': combined['Close'],
#     'High': combined['Close']*1.01,
#     'Low': combined['Close']*0.99,
#     'Close': combined['Close']
# }, index=combined.index)

# st.plotly_chart(plot_ohlc(combined_ohlc), use_container_width=True)

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# ---------------------------
# Utility Functions
# ---------------------------
def get_data(ticker, start_date):
    """Download stock OHLC data from yfinance"""
    stock_data = yf.download(ticker, start=start_date)
    return stock_data.dropna()

def stationary_check(series):
    series = series.dropna()
    adf_test = adfuller(series)
    return round(adf_test[1], 3)

def get_differencing_order(series):
    series = series.dropna()
    d = 0
    p_value = stationary_check(series)
    while p_value > 0.05:
        series = series.diff().dropna()
        p_value = stationary_check(series)
        d += 1
    return d

def fit_model(data, differencing_order, forecast_steps=30):
    model = ARIMA(data, order=(5, differencing_order, 0))
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=forecast_steps)
    return forecast.predicted_mean

def evaluate_model(data, differencing_order, test_size=30):
    train, test = data[:-test_size], data[-test_size:]
    predictions = fit_model(train, differencing_order, forecast_steps=test_size)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    return round(rmse, 2)

def get_forecast(data, differencing_order, forecast_steps=30):
    predictions = fit_model(data, differencing_order, forecast_steps=forecast_steps)
    start_date = datetime.now()
    dates = pd.date_range(start=start_date, periods=forecast_steps, freq='D')
    return pd.DataFrame(predictions, index=dates, columns=['Forecasted Close Price'])

def plot_forecast_table(forecast_df):
    header_color = 'blue'
    row_even_color = '#f8fafd'
    row_odd_color = '#e1efff'
    fig = go.Figure(
        data=[go.Table(
            header=dict(values=["<b>Date</b>"] + [f"<b>{col}</b>" for col in forecast_df.columns],
                        fill_color=header_color, line_color=header_color,
                        align="center", font=dict(color="white", size=12), height=35),
            cells=dict(values=[forecast_df.index.strftime('%Y-%m-%d').tolist()] +
                        [forecast_df[col].round(2).tolist() for col in forecast_df.columns],
                       fill_color=[[row_odd_color, row_even_color]*len(forecast_df)],
                       align="left", font=dict(color="darkslategray", size=12), height=30)
        )]
    )
    fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0))
    return fig

def plot_stock_chart(stock_df, forecast_df=None):
    """Candlestick chart with moving averages and optional forecast overlay"""
    stock_df['MA12'] = stock_df['Close'].rolling(12).mean()
    stock_df['MA26'] = stock_df['Close'].rolling(26).mean()

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=stock_df.index,
        open=stock_df['Open'],
        high=stock_df['High'],
        low=stock_df['Low'],
        close=stock_df['Close'],
        name='OHLC',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))

    # Moving averages
    fig.add_trace(go.Scatter(
        x=stock_df.index, y=stock_df['MA12'],
        mode='lines', name='MA12',
        line=dict(color='blue', width=2, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=stock_df.index, y=stock_df['MA26'],
        mode='lines', name='MA26',
        line=dict(color='orange', width=2, dash='dash')
    ))

    # Forecast line
    if forecast_df is not None:
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['Forecasted Close Price'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='purple', width=3),
            marker=dict(size=6, symbol='circle-open')
        ))

    fig.update_layout(
        title='Stock Price & Forecast',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        height=600,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="Stock Prediction", page_icon=":bar_chart:", layout="wide")
st.title("Stock Prediction Dashboard :bar_chart:")

col1, col2, col3 = st.columns(3)
today = datetime.today()

with col1:
    ticker = st.text_input("Enter Stock Ticker", value="AAPL")
with col2:
    start_date = st.date_input("Choose start date", datetime(today.year-1, today.month, today.day))
with col3:
    end_date = st.date_input("Choose end date", today)

# Fetch stock info
stock = yf.Ticker(ticker)
st.subheader(ticker)
st.write(stock.info.get('longBusinessSummary', 'No description available'))
st.write("Sector:", stock.info.get('sector', 'N/A'))
st.write("Industry:", stock.info.get('industry', 'N/A'))
st.write("Market Cap:", stock.info.get('marketCap', 'N/A'))
st.write("Website:", stock.info.get('website', 'N/A'))

# Fetch OHLC data
stock_data = get_data(ticker, start_date)

# Differencing order and RMSE
d_order = get_differencing_order(stock_data['Close'])
rmse = evaluate_model(stock_data['Close'].values, d_order)
st.write('Model RMSE score:', rmse)

# Forecast
forecast_df = get_forecast(stock_data['Close'].values, d_order)
st.subheader("Forecast for next 30 days")
st.plotly_chart(plot_forecast_table(forecast_df), use_container_width=True)

# Beautiful stock chart with forecast overlay
st.plotly_chart(plot_stock_chart(stock_data, forecast_df), use_container_width=True)
