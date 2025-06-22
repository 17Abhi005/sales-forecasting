import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def forecast_sales(df, periods, model_type="Prophet"):
    df = df.rename(columns={"Order Date": "ds", "Sales": "y"})
    df = df[["ds", "y"]].groupby("ds").sum().reset_index()
    df = df.sort_values("ds")

    if model_type == "Prophet":
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        forecast_df = forecast[["ds", "yhat"]]
    else:  # ARIMA
        df.set_index("ds", inplace=True)
        model = ARIMA(df["y"], order=(5, 1, 0))
        model_fit = model.fit()
        forecast_values = model_fit.forecast(steps=periods)
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=periods)
        forecast_df = pd.DataFrame({"ds": future_dates, "yhat": forecast_values.values})

    # Accuracy metrics using last known 7 days
    if len(df) >= 14:
        train = df[:-7]
        test = df[-7:]
        if model_type == "Prophet":
            m = Prophet()
            m.fit(train)
            f = m.predict(test[["ds"]])
            preds = f["yhat"].values
        else:
            m = ARIMA(train["y"], order=(5, 1, 0))
            m_fit = m.fit()
            preds = m_fit.forecast(steps=7)
        rmse = np.sqrt(mean_squared_error(test["y"], preds))
        mae = mean_absolute_error(test["y"], preds)
    else:
        rmse = mae = None

    return forecast_df, rmse, mae
