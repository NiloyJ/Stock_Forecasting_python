# import plotly.graph_objects as go
# import datautil 
# import pandas_ta as pd
# import datetime 

# def plotly_table(dataframe):
#     headerColor = 'grey'
#     rowEventColor = 'lightgrey'
#     rowOddColor = 'white'
#     fig = go.Figure(data=[go.Table(header=dict(
#         values=["<b><b>"]+["<b>"+str(i)[:10]+"<b>" for i in dataframe.columns],
#         line_color='darkslategray', fill_color='#0078ff', align='center', font=dict(color='white', size=12), height=35),

#     ),
#     cells=dict(
#         values=[["<b>"]+str(i)+"<b>" for i in dataframe.index]] + [dataframe[i] for i in dataframe.columns], fillColor=[[rowOddColor,rowEventColor]*len(dataframe)],
#         align='left', line_color=['white'], font=dict(color='darkslategray', size=11), height=30)
#     ))
#     ])
# fig.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0)))
# return fig

import plotly.graph_objects as go
import pandas as pd

def plotly_table(dataframe: pd.DataFrame):
    header_color = 'blue'
    row_even_color = '#f8fafd'
    row_odd_color = '#e1efff'

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["<b></b>"] + [f"<b>{col}</b>" for col in dataframe.columns],
                    fill_color='#0078ff',
                    line_color='#0078ff',
                    align="center",
                    font=dict(color="white", size=12),
                    height=35
                ),
                cells=dict(
                    values=[
                        dataframe.index.astype(str).tolist()
                    ] + [
                        dataframe[col].tolist() for col in dataframe.columns
                    ],
                    fill_color=[
                        [row_odd_color, row_even_color] * len(dataframe)
                    ],
                    align="left",
                    font=dict(color="darkslategray", size=20),
                    height=30
                )
            )
        ]
    )

    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return fig

# def filter_data(dataframe, num_period):
#     if num_period == '1 Month':
#         date = dataframe.index[-1] + dateutil.relativedelta.relativedelta(months=-1)
#     elif num_period == '5d':
#         date = dataframe.index[-1] + dateutil.relativedelta.relativedelta(days=-5)
#     elif num_period=='6mo':
#         date = dataframe.index[-1] + dateutil.relativedelta.relativedelta(months=-6)
#     elif num_period=='1y':
#         date = dataframe.index[-1] + dateutil.relativedelta.relativedelta(years=-1)
#     elif num_period=='5y':
#         date = dataframe.index[-1] + dateutil.relativedelta.relativedelta(years=-5)
#     elif num_period=='ytd':
#         date = datetime.datetime(dataframe.index[-1].year, 1, 1).strftime('%Y-%m-%d')
#     else:
#         date = dataframe.index[0]

#     return dataframe.reset_index()[dataframe.reset_index()['Date']>date]

# def close_chart(dataframe, num_period=False):
#     if num_period:
#         dataframe = filter_data(dataframe, num_period)
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Open'], mode='lines', name='Open', line=dict(width=2, color='#5ab7ff')))
#     fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Close'], mode='lines', name='Close', line=dict(width=2, color='#ff6f61')))

#     fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['High'], mode='lines', name='High', line=dict(width=1, color='#90ee90', dash='dash')))
#     fig.add_trace(go.scatter(x=dataframe['Date'], y=dataframe['Low'], mode='lines', name='Low', line=dict(width=1, color='#ffa500', dash='dash')))
    
#     fig.update_xaxes(rangeslider_visible=True)
#     fig.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor='white', paper_bgcolor='#eleff',legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
#     return fig

# def candlesstick(datarame, num_period):
#     datarame = filter_data(datarame, num_period)
#     fig = go.figure()

#     fig.add_trace(go.Candlestick(x=datarame['Date'], open=dataframe['Open'], high=dataframe['High'], low=dataframe['Low'], close=dataframe['Close'], increasing_line_color='#5ab7ff', decreasing_line_color='#ff6f61'))

#     fig.update_xaxes(showlegend=True), height=500, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor='white', paper_bgcolor='#eleff',legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
#     return fig 

# def RSI(dataframe, num_period):
#     dataframe["RSI"] = pta.rsi(dataframe['Close'])
#     dataframe = filter_data(dataframe, num_period)
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['RSI'], mode='lines', name='RSI', line=dict(width=2, color='#5ab7ff')))
#     fig.add_trace(go.scatter(x=dataframe['Date'], y=[70]*len(dataframe), mode='lines', name='Overbought', line=dict(width=1, color='#ff6f61', dash='dash')))

#     fig.add_trace(go.scatter(x=dataframe['Date'], y=[30]*len(dataframe), mode='lines', name='Oversold', line=dict(width=1, color='#90ee90', dash='dash')))

#     fig.update_layout(yaxis_range=[0,100]), height=200, plot_bgcolor='white', paper_bgcolor='#eleff',legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
#     return fig

