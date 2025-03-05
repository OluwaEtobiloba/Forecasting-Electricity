# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 09:16:15 2024

@author: Oluwatobiloba Alao
"""

pip install arch
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import statsmodels.api as sm
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose


d1 = pd.read_csv('C:/Users/user/Desktop/DS 480/DTS.csv')
d1.head()


#for the purpose of the model fittings i would be focusing on the Total usage of electricity 
usage_series = d1['USAGE'].replace([np.inf, -np.inf], np.nan).dropna() # Dependent variable

#Plot the ACF and PACF plots to determine the p and q parameters for ARIMA order.
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plot_acf(usage_series, lags=40, ax=plt.gca())
plt.title('Autocorrelation Function (ACF)')
plt.subplot(1, 2, 2)
plot_pacf(usage_series, lags=40, ax=plt.gca(), method='ywm')
plt.title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()


# Convert 'DATE' to datetime and set as index
d1['DATE'] = pd.to_datetime(d1['DATE'], errors='coerce')
d1.set_index('DATE', inplace=True)


# Fit an ARIMA(1, 1, 1) model as a starting point
arima_model= ARIMA(usage_series, order=(1, 1, 1))
arima_result = arima_model.fit()
arima_result.summary()


# Plot the original series and the ARIMA model's fitted values
plt.figure(figsize=(12, 6))
plt.plot(d1.index, usage_series, label='Original usage Data', color='blue')
plt.plot(d1.index, arima_result.fittedvalues, label='ARIMA Fitted Values', color='red')
plt.title('ARIMA Model Fitting on Electric Usage')
plt.xlabel('Date')
plt.ylabel('Usage')
plt.legend()
plt.grid(True)
plt.show()

# Forecast for the next 30 days for ARIMA
forecast_steps = 30
forecast_result = arima_result.forecast(steps=forecast_steps)

# Create a range of dates for the forecast
forecast_dates = pd.date_range(start=usage_series.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')

# Convert forecast to a Pandas Series
forecast_series_usage = pd.Series(forecast_result, index=forecast_dates)
print(forecast_series_usage)
#Plot the original series along with the forecast
plt.figure(figsize=(12, 6))

# Plot the historical data
plt.plot(usage_series, label='Historical Usage', color='blue')

# Plot the forecasted data
plt.plot(forecast_series_usage, label='Forecast', color='green')

# Adding title, labels, and legend
plt.title('ARIMA Model Forecast for Daily Electric Usage')
plt.xlabel('Date')
plt.ylabel('Total Usage (kWh)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()



# Fit a SARIMA model to the Usage data
# We'll use SARIMA(1, 1, 1)(1, 1, 1, 7) to account for potential seasonality (assuming daily seasonality)
sarima_model = SARIMAX(usage_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
sarima_result = sarima_model.fit()
sarima_result.summary()

plt.figure(figsize=(12, 6))
plt.plot(d1.index, usage_series, label='Original usage Data', color='blue')
plt.plot(d1.index, sarima_result.fittedvalues, label='SARIMA Fitted Values', color='green')
plt.title('SARIMA Model Fitting on Electric Usage')
plt.xlabel('Year')
plt.ylabel('Revenue($B)')
plt.legend()
plt.grid(True)
plt.show()


# Decompose the time series to extract trend, seasonal, and residual components
# Assuming weekly seasonality (period=7)
decomposition = seasonal_decompose(usage_series, model='additive', period=7)

# Plot the decomposition results: observed, trend, seasonal, and residual components
plt.figure(figsize=(5, 5))

# Observed data
plt.subplot(4, 1, 1)
plt.plot(decomposition.observed, label='Observed', color='blue')
plt.title('Observed')
plt.grid(True)

# Trend component
plt.subplot(4, 1, 2)
plt.plot(decomposition.trend, label='Trend', color='green')
plt.title('Trend')
plt.grid(True)

# Seasonal component
plt.subplot(4, 1, 3)
plt.plot(decomposition.seasonal, label='Seasonal', color='orange')
plt.title('Seasonal')
plt.grid(True)

# Residual component
plt.subplot(4, 1, 4)
plt.plot(decomposition.resid, label='Residual', color='red')
plt.title('Residual')
plt.grid(True)

plt.tight_layout()
plt.show()



# Forecasting for the next 30 days using the adjusted SARIMA model
forecast_steps_sarima = 30
forecast_result_sarima = sarima_result.get_forecast(steps=forecast_steps_sarima)

# Extract forecast mean
forecast_mean_sarima = forecast_result_sarima.predicted_mean

# Create a range of dates for the forecast
forecast_dates_sarima = pd.date_range(start=d1.index[-1] + pd.Timedelta(days=1), periods=forecast_steps_sarima, freq='D')

# Convert forecast to a Pandas Series
forecast_series_sarima = pd.Series(forecast_mean_sarima, index=forecast_dates_sarima)

# Plot the original series along with the adjusted SARIMA forecast
plt.figure(figsize=(12, 6))

# Plot the historical data
plt.plot(usage_series, label='Historical Usage', color='blue')

# Plot the forecasted data with the adjusted SARIMA model
plt.plot(forecast_series_sarima, label='SARIMA Forecast', color='orange', linestyle='--')

# Adding title, labels, and legend
plt.title('SARIMA Model Forecast for Daily Electric Usage')
plt.xlabel('Date')
plt.ylabel('Usage (kWh)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()



# Calculate the percentage change in usage to model volatility
usage_diff = d1['USAGE'].pct_change().replace([np.inf, -np.inf], np.nan).dropna()


# Fit an ARCH model to the percentage changes
arch_model_fun = arch_model(usage_diff, vol='ARCH', p=1)
arch_result = arch_model_fun.fit(disp='off')
arch_result.summary()


# Plot the original percentage changes and the ARCH model's conditional volatility
plt.figure(figsize=(12, 6))
plt.plot(usage_diff.index, usage_diff, label='Total Usage', color='blue')
plt.plot(usage_diff.index, arch_result.conditional_volatility, label='ARCH Conditional Volatility', color='orange')
plt.title('ARCH Model Fitting on Electric Usage Changes')
plt.xlabel('Date')
plt.ylabel('Usage')
plt.legend()
plt.grid(True)
plt.show()


# Forecasting with the fitted ARCH model for the next 30 periods
forecast_steps_arch = 30
arch_forecast = arch_result.forecast(horizon=forecast_steps_arch)

# Extract the forecast of the conditional volatility
forecast_cond_vol = arch_forecast.variance.values[-1]

# Create a range of dates for the forecast
forecast_dates_arch = pd.date_range(start=usage_diff.index[-1] + pd.Timedelta(days=1), periods=forecast_steps_arch, freq='D')

# Convert forecast to a Pandas Series
forecast_series_arch = pd.Series(forecast_cond_vol, index=forecast_dates_arch)

# Plot the original percentage changes and the ARCH model's forecasted conditional volatility
plt.figure(figsize=(12, 6))

# Plot the historical percentage changes in usage
plt.plot(usage_diff.index, usage_diff, label='Usage Change (%)', color='blue')

# Plot the forecasted conditional volatility
plt.plot(forecast_series_arch, label='ARCH Forecasted Conditional Volatility', color='red', linestyle='--')

# Adding title, labels, and legend
plt.title('ARCH Model Forecast for Electric Usage Changes')
plt.xlabel('Date')
plt.ylabel('Usage Volatility')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()

#fit a GARCH Model to the percentage change 
garch_model_fun= arch_model(usage_diff, vol='Garch', p=1, q=1)
garch_result = garch_model_fun.fit(disp='off')
print(garch_result.summary())

# Plot the original percentage changes and the GARCH model's conditional volatility
plt.figure(figsize=(12, 6))
plt.plot(usage_diff.index, usage_diff, label='Revenue Change (%)', color='blue')
plt.plot(usage_diff.index, garch_result.conditional_volatility, label='GARCH Conditional Volatility', color='red')
plt.title('GARCH Model Fitting on Electric Usage Changes')
plt.xlabel('Date')
plt.ylabel('Usage')
plt.legend()
plt.grid(True)
plt.show()

# Forecasting with the fitted GARCH model for the next 30 periods
forecast_steps_garch = 30
garch_forecast = garch_result.forecast(horizon=forecast_steps_garch)

# Extract the forecast of the conditional volatility
forecast_cond_vol = garch_forecast.variance.values[-1]

# Create a range of dates for the forecast
forecast_dates_garch = pd.date_range(start=usage_diff.index[-1] + pd.Timedelta(days=1), periods=forecast_steps_garch, freq='D')

# Convert forecast to a Pandas Series
forecast_series_garch = pd.Series(forecast_cond_vol, index=forecast_dates_garch)

# Plot the original percentage changes and the ARCH model's forecasted conditional volatility
plt.figure(figsize=(12, 6))

# Plot the historical percentage changes in usage
plt.plot(usage_diff.index, usage_diff, label='Usage Change (%)', color='blue')

# Plot the forecasted conditional volatility
plt.plot(forecast_series_garch, label='GARCH Forecasted Conditional Volatility', color='red', linestyle='--')

# Adding title, labels, and legend
plt.title('GARCH Model Forecast for Electric Usage Changes')
plt.xlabel('Date')
plt.ylabel('Usage Volatility')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()



