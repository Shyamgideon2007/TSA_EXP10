# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL


### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

data = pd.read_csv('Temperature.csv')

data['date'] = pd.to_datetime(data['date'])

plt.plot(data['date'], data['temp'])
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Temperature Time Series')
plt.show()

def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')

    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

check_stationarity(data['temp'])

plot_acf(data['temp'])
plt.show()

plot_pacf(data['temp'])
plt.show()

sarima_model = SARIMAX(
    data['temp'],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)

sarima_result = sarima_model.fit()

train_size = int(len(data) * 0.8)

train = data['temp'][:train_size]
test = data['temp'][train_size:]

sarima_model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)

sarima_result = sarima_model.fit()

predictions = sarima_result.predict(
    start=len(train),
    end=len(train) + len(test) - 1
)

mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)

print('RMSE:', rmse)

plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')

plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('SARIMA Model Predictions')

plt.legend()
plt.show()
```
### OUTPUT:
<img width="562" height="455" alt="image" src="https://github.com/user-attachments/assets/1529af5a-eb04-4396-9996-b88654ce323e" />

<img width="568" height="435" alt="image" src="https://github.com/user-attachments/assets/186c8829-4559-4098-a9b9-d0e429ba3230" />

<img width="562" height="455" alt="image" src="https://github.com/user-attachments/assets/5604d217-e93d-4608-973b-473b2924ea6c" />

### RESULT:
Thus the program run successfully based on the SARIMA model.
