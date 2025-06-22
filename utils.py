import pandas as pd
import streamlit as st


@st.cache_data
def load_data(path):
    df = pd.read_csv(path, parse_dates=["Order Date"], encoding='latin1')
    return df


def filter_data(df, region, category, date_range):
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    mask = (
        ((df["Region"] == region) if region else True) &
        ((df["Category"] == category) if category else True) &
        (df["Order Date"] >= start_date) &
        (df["Order Date"] <= end_date)
    )
    return df.loc[mask]

