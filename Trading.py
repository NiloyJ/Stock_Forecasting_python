import streamlit as st

st.title("Trading Page")

st.set_page_config(
    page_title="Trading Page",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)



st.title("Trading Page with whatever you want to add :bar_chart:")

st.header("A platform to look at all information before betting your bucks")

st.image("trading.jpg")

st.markdown("# Our horizon includes :")

st.markdown("### :one: Real time equity information")
st.write("Get up-to-date data on stock prices, market trends, and financial news to make informed trading decisions.")

st.markdown("### :two: Equity predictions")
st.write("Predicting the closing price")

st.markdown("### :three: CAPM Return")
st.write("Calculating expected returns based on the Capital Asset Pricing Model (CAPM) to help you assess investment risks and returns.")

st.markdown("### :four: CAPM Beta")
st.write("Understanding the volatility of a stock in relation to the overall market using CAPM Beta.")

