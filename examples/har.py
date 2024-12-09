import pandas as pd
import numpy as np
import statsmodels.api as sm


def compute_daily_rv(returns):
    """
    Compute daily realized volatility
    """
    return abs(returns)  # For daily data, we can use absolute returns as a simple volatility measure


def har_model(returns, lags=(1, 5, 22)):
    """
    Fit a HAR model to daily returns data

    :param returns: pandas Series of daily returns
    :param lags: tuple of (daily, weekly, monthly) lags
    :return: model results and prepared data
    """
    # Compute daily RV
    rv = compute_daily_rv(returns)

    # Create lagged terms
    rv_d = rv.shift(1)  # Daily (t-1)
    rv_w = rv.rolling(window=5).mean().shift(1)  # Weekly average (t-5 to t-1)
    rv_m = rv.rolling(window=22).mean().shift(1)  # Monthly average (t-22 to t-1)

    # Combine into a dataframe
    data = pd.DataFrame({
        'RV': rv,
        'RV_D': rv_d,
        'RV_W': rv_w,
        'RV_M': rv_m
    }).dropna()

    # Fit the HAR model
    X = sm.add_constant(data[['RV_D', 'RV_W', 'RV_M']])
    y = data['RV']
    model = sm.OLS(y, X).fit()

    return model, data


def forecast_volatility(model, last_data, horizon=200):
    """
    Generate volatility forecasts

    :param model: fitted HAR model
    :param last_data: last observed values for RV_D, RV_W, and RV_M
    :param horizon: number of days to forecast
    :return: array of forecasted volatilities
    """
    forecasts = np.zeros(horizon)

    # Get model parameters
    beta_const = model.params[0]
    beta_d = model.params[1]
    beta_w = model.params[2]
    beta_m = model.params[3]

    # Initialize with last known values
    last_d = last_data['RV_D'].iloc[-1]
    last_w = last_data['RV_W'].iloc[-1]
    last_m = last_data['RV_M'].iloc[-1]

    # Generate forecasts
    for h in range(horizon):
        # Compute forecast for day h+1
        forecast = beta_const + beta_d * last_d + beta_w * last_w + beta_m * last_m
        forecasts[h] = forecast

        # Update lagged values
        if h >= 1:
            last_d = forecasts[h - 1]
        if h >= 5:
            last_w = np.mean(forecasts[max(0, h - 5):h])
        if h >= 22:
            last_m = np.mean(forecasts[max(0, h - 22):h])

    return forecasts


# Example usage
if __name__ == "__main__":
    # Generate example data
    np.random.seed(42)
    n = 1000
    returns = pd.Series(
        np.random.normal(0, 0.01, size=n),
        index=pd.date_range('2023-01-01', periods=n)
    )

    # Fit the HAR model
    har_results, data = har_model(returns)
    print("\nModel Summary:")
    print(har_results.summary())

    # Generate forecasts
    forecasts = forecast_volatility(har_results, data, horizon=200)

    # Create forecast DataFrame with dates
    last_date = returns.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=200)
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Predicted_Volatility': forecasts
    })

    print("\nVolatility Forecasts (first 10 days):")
    print(forecast_df.head(10))

    # Optional: Plot the forecasts
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.plot(forecast_df['Date'], forecast_df['Predicted_Volatility'])
        plt.title('HAR Volatility Forecasts - Next 200 Days')
        plt.xlabel('Date')
        plt.ylabel('Predicted Volatility')
        plt.grid(True)
        plt.show()
    except ImportError:
        print("Matplotlib not installed. Skipping plot.")
