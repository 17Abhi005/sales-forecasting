import streamlit as st
import pandas as pd
import plotly.express as px
from forecast import forecast_sales
from utils import load_data, filter_data

st.set_page_config(layout="wide")
st.title("E-commerce Sales Forecasting Dashboard")

# Load data
df = load_data("data/superstore.csv")

# Sidebar Filters
regions = sorted(df["Region"].dropna().unique())
categories = sorted(df["Category"].dropna().unique())
min_date, max_date = df["Order Date"].min(), df["Order Date"].max()

region = st.sidebar.selectbox("Select Region", [""] + regions)
category = st.sidebar.selectbox("Select Category", [""] + categories)
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])

# Filtered data
filtered_df = filter_data(df, region, category, date_range)

# Display Raw Data
with st.expander("Show Raw Data"):
    st.write(filtered_df)

# Time Series Plot
sales_ts = filtered_df.groupby("Order Date")["Sales"].sum().reset_index()
fig = px.line(sales_ts, x="Order Date", y="Sales", title="Daily Sales Trend")
st.plotly_chart(fig, use_container_width=True)

# Exploratory Charts
st.subheader("Exploratory Sales Breakdown")

col1, col2 = st.columns(2)

with col1:
    region_sales = filtered_df.groupby("Region")["Sales"].sum().reset_index()
    fig_region = px.pie(region_sales, names="Region", values="Sales", title="Sales by Region")
    st.plotly_chart(fig_region, use_container_width=True)

with col2:
    category_sales = filtered_df.groupby("Category")["Sales"].sum().reset_index()
    fig_category = px.bar(category_sales, x="Category", y="Sales", title="Sales by Category")
    st.plotly_chart(fig_category, use_container_width=True)

# Model selection
model_choice = st.sidebar.radio("Choose Forecast Model", ["Prophet", "ARIMA"])

# Forecasting
if st.button("Generate Forecast"):
    forecast_df, rmse, mae = forecast_sales(filtered_df, periods=30, model_type=model_choice)

    st.subheader(f"{model_choice} Forecast (Next 30 Days)")
    fig_forecast = px.line(forecast_df, x="ds", y="yhat", title=f"{model_choice} Forecast")
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Show accuracy metrics
    if rmse is not None:
        st.success(f"RMSE: {rmse:.2f} | MAE: {mae:.2f}")
    else:
        st.warning("Not enough data to compute accuracy metrics.")

    # Downloadable CSV
    csv_data = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Forecast Report", 
        csv_data, 
        "forecast_report.csv", 
        "text/csv", 
        key=f"download_button_{model_choice}"
    )
