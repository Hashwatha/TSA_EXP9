# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
## AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
## ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
## PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("cardekho.csv")

print("Columns in dataset:", data.columns)
print(data.head())

data['year'] = pd.to_datetime(data['year'], format='%Y')
data.set_index('year', inplace=True)

data_yearly = data.groupby('year')['selling_price'].mean().reset_index()
data_yearly.set_index('year', inplace=True)

plt.figure(figsize=(10, 5))
plt.plot(data_yearly.index, data_yearly['selling_price'], marker='o', label='Selling Price')
plt.title('Average Selling Price per Year (Cardekho Dataset)')
plt.xlabel('Year')
plt.ylabel('Selling Price')
plt.grid()
plt.legend()
plt.show()

def arima_model(data, target_variable, order=(5, 1, 2)):
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()

    forecast = fitted_model.forecast(steps=len(test_data))

    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))

    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data')
    plt.plot(test_data.index, forecast, label='Forecasted Data', color='red')
    plt.xlabel('Year')
    plt.ylabel(target_variable)
    plt.title(f'ARIMA Forecasting for {target_variable}')
    plt.legend()
    plt.grid()
    plt.show()

    print("Root Mean Squared Error (RMSE):", round(rmse, 3))
    return fitted_model, forecast

model, forecast = arima_model(data_yearly, 'selling_price', order=(5, 1, 2))

```
## OUTPUT:

<img width="1258" height="657" alt="image" src="https://github.com/user-attachments/assets/0cde2288-bce7-45e6-b18d-b777550dd1e1" />

<img width="1227" height="732" alt="image" src="https://github.com/user-attachments/assets/a7d4e793-7d99-4f4f-b025-81f5b1b46c6d" />

## RESULT:
Thus the program run successfully based on the ARIMA model using python.
