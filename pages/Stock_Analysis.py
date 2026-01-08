# import streamlit as st
# import pandas as pd 
# import yfinance as yf
# import plotly.graph_objects as go
# import datetime 
# from pages.utils.plotly_figure import plotly_table

# st.set_page_config(
#     page_title="Stock Prediction Page",
#     page_icon=":bar_chart:",
#     layout="wide"
# )

# st.title("Stock Prediction Page :bar_chart:")

# col1, col2, col3 = st.columns(3)

# today = datetime.date.today()

# with col1:
#     ticker = st.text_input("Enter Stock Ticker", value="TSLA")
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
#     fig_df = plotly_table(df)
#     st.plotly_chart(fig_df, use_container_width=True)

# with col2:
#     df = pd.DataFrame(index = ['Quick Ratio', 'Revenue per share', 'Profit Margins', 'Debt to Equity', 'Return on Equity'])
#     df[''] = [stock.info['quickRatio'], stock.info['revenuePerShare'],  stock.info['profitMargins'], stock.info['debtToEquity'], stock.info['returnOnEquity']]
#     fig_df = plotly_table(df)
#     st.plotly_chart(fig_df, use_container_width=True)

# data = yf.download(ticker, start=start_date, end=end_date)

# col1, col2, col3 = st.columns(3)
# daily_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
# col1.metric("Daily Change", str(round(data['Close'].iloc[-1],2)), str(round(daily_change,2)))

# last_10_df = data.tail(10).sort_index(ascending=False).round(3)
# fig_df = plotly_table(last_10_df)
# st.write('Historical data of last 10 days')
# st.plotly_chart(fig_df, use_container_width=True)

# # col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12 = st.columns([1,1,1,1,1,1,1,1,1,1,1])

# col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12 = st.columns([1,1,1,1,1,1,1,1,1,1,1,1])


# num_period = ''

# with col1:
#     if st.button('5D'):
#         num_period = '5D'
# with col2:
#     if st.button('1M'):
#         num_period = '1mo'
# with col3:
#     if st.button('6M'):
#         num_period = '6mo'
# with col4:
#     if st.button('YTD'):
#         num_period = 'ytd'  
# with col5: 
#     if st.button('1Y'):
#         num_period = '1y'
# with col6:
#     if st.button('5Y'):
#         num_period = '5y'
# with col7:
#     if st.button('MAX'):
#         num_period = 'max'

# col1, col2, col3 = st.columns([1,1,4])
# with col1:
#     chart_type = st.selectbox('', {'Candle', 'Line'})
# with col2:
#     if chart_type == 'Candle':
#         indicators = st.selectbox('',('RSI', 'MACD'))
#     else:
#         indicators = st.selectbox('',('RSI', 'Moving Average', 'MACD'))

# ticker = yf.Ticker(ticker)
# new_df1 = ticker.history(period='max')
# data1 = ticker.history(period='max')
# if num_period == '':
#     if chart_type=='Candle' and indicators == 'RSI':
#         st.plotly_chart(candlestick(data1, '1y'), use_container_width=True)
#         st.plotly_chart(RSI(data1, '1y'), use_container_width=True)
    
#     if chart_type=='Candle' and indicators == 'MACD':
#         st.plotly_chart(candlestick(data1, '1y'), use_container_width=True)
#         st.plotly_chart(MACD(data1, '1y'), use_container_width=True)

#     if chart_type=='Line' and indicators == 'RSI':
#         st.plotly_chart(line_chart(data1, '1y'), use_container_width=True)
#         st.plotly_chart(RSI(data1, '1y'), use_container_width=True)

#     if chart_type=='Line' and indicators == 'Moving Average':
#         st.plotly_chart(line_chart(data1, '1y'), use_container_width=True)
#         st.plotly_chart(Moving_Average(data1, '1y'), use_container_width=True)

#     if chart_type=='Line' and indicators == 'MACD':
#         st.plotly_chart(line_chart(data1, '1y'), use_container_width=True)
#         st.plotly_chart(MACD(data1, '1y'), use_container_width=True)


import streamlit as st
import pandas as pd 
import yfinance as yf
import plotly.graph_objects as go
import datetime 
import numpy as np
from pages.utils.plotly_figure import plotly_table

st.set_page_config(
    page_title="Stock Prediction Page",
    page_icon=":bar_chart:",
    layout="wide"
)

st.title("Stock Prediction Page :bar_chart:")

# ========== CHART FUNCTIONS ==========

def candlestick(data, period):
    """Create candlestick chart"""
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlestick'
    )])
    fig.update_layout(
        title=f'Candlestick Chart ({period})',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=500
    )
    return fig

def line_chart(data, period):
    """Create line chart"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue')
    ))
    fig.update_layout(
        title=f'Line Chart ({period})',
        xaxis_title='Date',
        yaxis_title='Price',
        height=500
    )
    return fig

def RSI(data, period, window=14):
    """Calculate and plot RSI"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=rsi,
        mode='lines',
        name='RSI',
        line=dict(color='purple')
    ))
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5)
    fig.update_layout(
        title=f'RSI ({period})',
        xaxis_title='Date',
        yaxis_title='RSI Value',
        height=300,
        yaxis_range=[0, 100]
    )
    return fig

def MACD(data, period):
    """Calculate and plot MACD"""
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=macd,
        mode='lines',
        name='MACD',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=data.index,
        y=signal,
        mode='lines',
        name='Signal',
        line=dict(color='red')
    ))
    fig.update_layout(
        title=f'MACD ({period})',
        xaxis_title='Date',
        yaxis_title='MACD Value',
        height=300
    )
    return fig

def Moving_Average(data, period):
    """Plot moving averages"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='lightgray')
    ))
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'].rolling(window=20).mean(),
        mode='lines',
        name='20-day MA',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'].rolling(window=50).mean(),
        mode='lines',
        name='50-day MA',
        line=dict(color='red')
    ))
    fig.update_layout(
        title=f'Moving Averages ({period})',
        xaxis_title='Date',
        yaxis_title='Price',
        height=500
    )
    return fig

# ========== MAIN APP CODE ==========

col1, col2, col3 = st.columns(3)

today = datetime.date.today()

with col1:
    ticker = st.text_input("Enter Stock Ticker", value="TSLA")
with col2:
    start_date = st.date_input("Choose start date", datetime.date(today.year - 1, today.month, today.day))  
with col3: 
    end_date = st.date_input("Choose end date", datetime.date(today.year, today.month, today.day))  

st.subheader(ticker)

stock = yf.Ticker(ticker)

st.write(stock.info['longBusinessSummary'])
st.write(stock.info['sector'])
st.write(stock.info['industry'])
st.write("Market Cap:", stock.info['marketCap'])
st.write(stock.info['website'])

col1, col2 = st.columns(2)

with col1:
    df = pd.DataFrame(index=['Market cap', 'Beta', 'EPS', 'PE ratio'])
    df[''] = [stock.info['marketCap'], stock.info['beta'], stock.info['trailingEps'], stock.info['trailingPE']]
    fig_df = plotly_table(df)
    st.plotly_chart(fig_df, use_container_width=True)

with col2:
    df = pd.DataFrame(index=['Quick Ratio', 'Revenue per share', 'Profit Margins', 'Debt to Equity', 'Return on Equity'])
    df[''] = [stock.info['quickRatio'], stock.info['revenuePerShare'], stock.info['profitMargins'], stock.info['debtToEquity'], stock.info['returnOnEquity']]
    fig_df = plotly_table(df)
    st.plotly_chart(fig_df, use_container_width=True)

data = yf.download(ticker, start=start_date, end=end_date)

col1, col2, col3 = st.columns(3)
daily_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
col1.metric("Daily Change", str(round(data['Close'].iloc[-1],2)), str(round(daily_change,2)))

last_10_df = data.tail(10).sort_index(ascending=False).round(3)
fig_df = plotly_table(last_10_df)
st.write('Historical data of last 10 days')
st.plotly_chart(fig_df, use_container_width=True)

col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12 = st.columns([1,1,1,1,1,1,1,1,1,1,1,1])

num_period = ''

with col1:
    if st.button('5D'):
        num_period = '5D'
with col2:
    if st.button('1M'):
        num_period = '1mo'
with col3:
    if st.button('6M'):
        num_period = '6mo'
with col4:
    if st.button('YTD'):
        num_period = 'ytd'  
with col5: 
    if st.button('1Y'):
        num_period = '1y'
with col6:
    if st.button('5Y'):
        num_period = '5y'
with col7:
    if st.button('MAX'):
        num_period = 'max'

col1, col2, col3 = st.columns([1,1,4])
with col1:
    chart_type = st.selectbox('', ['Candle', 'Line'])
with col2:
    if chart_type == 'Candle':
        indicators = st.selectbox('', ['RSI', 'MACD'])
    else:
        indicators = st.selectbox('', ['RSI', 'Moving Average', 'MACD'])

ticker_obj = yf.Ticker(ticker)
data1 = ticker_obj.history(period='max')

# Filter data based on selected period
if num_period != '':
    if num_period == '5D':
        data1 = data1.tail(5)
    elif num_period == '1mo':
        data1 = data1.tail(30)
    elif num_period == '6mo':
        data1 = data1.tail(180)
    elif num_period == 'ytd':
        current_year = datetime.datetime.now().year
        data1 = data1[data1.index.year == current_year]
    elif num_period == '1y':
        data1 = data1.tail(365)
    elif num_period == '5y':
        data1 = data1.tail(365*5)
    # 'max' already handled by using full data

# Display charts based on selection
if chart_type == 'Candle' and indicators == 'RSI':
    st.plotly_chart(candlestick(data1, num_period if num_period else 'MAX'), use_container_width=True)
    st.plotly_chart(RSI(data1, num_period if num_period else 'MAX'), use_container_width=True)

elif chart_type == 'Candle' and indicators == 'MACD':
    st.plotly_chart(candlestick(data1, num_period if num_period else 'MAX'), use_container_width=True)
    st.plotly_chart(MACD(data1, num_period if num_period else 'MAX'), use_container_width=True)

elif chart_type == 'Line' and indicators == 'RSI':
    st.plotly_chart(line_chart(data1, num_period if num_period else 'MAX'), use_container_width=True)
    st.plotly_chart(RSI(data1, num_period if num_period else 'MAX'), use_container_width=True)

elif chart_type == 'Line' and indicators == 'Moving Average':
    st.plotly_chart(line_chart(data1, num_period if num_period else 'MAX'), use_container_width=True)
    st.plotly_chart(Moving_Average(data1, num_period if num_period else 'MAX'), use_container_width=True)

elif chart_type == 'Line' and indicators == 'MACD':
    st.plotly_chart(line_chart(data1, num_period if num_period else 'MAX'), use_container_width=True)
    st.plotly_chart(MACD(data1, num_period if num_period else 'MAX'), use_container_width=True)