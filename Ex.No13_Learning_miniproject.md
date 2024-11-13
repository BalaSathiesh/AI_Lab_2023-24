# Ex.No: 13 Machine Learning – Use Supervised Learning  
### DATE:                                                                            
### REGISTER NUMBER :212222040022 
### AIM: 
To write a program to train the classifier for Stock Price Prediction.
###  Algorithm:
1.Load historical stock price data and scale it between 0 and 1 for better model performance.
2.Split the scaled data into training and testing sets (e.g., 80% training, 20% testing).
3,Define a function to create sequences of stock prices, with each sequence representing a time window for training.
4.Build an LSTM neural network model with dropout layers to prevent overfitting and train it on the prepared data.
5.Use the trained model to predict stock prices and scale them back to their original range.
6.Plot the actual vs. predicted stock prices to visually evaluate the model’s performance.
### Program:
```
!pip install yfinance
!pip install tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
stock_symbol = 'AAPL'  # Apple Inc. example
data = yf.download(stock_symbol, start="2015-01-01", end="2023-01-01")
data = data[['Close']]
data.head()
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)


train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]


def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)


time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))


model.compile(optimizer='adam', loss='mean_squared_error')


history = model.fit(X_train, y_train, batch_size=64, epochs=50, validation_data=(X_test, y_test))
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))


model.compile(optimizer='adam', loss='mean_squared_error')


history = model.fit(X_train, y_train, batch_size=64, epochs=50, validation_data=(X_test, y_test))
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

train_data_length = train_size + time_step
plt.figure(figsize=(16,8))


plt.plot(data['Close'], label='Actual Prices')
plt.plot(range(time_step, train_size), train_predict, label='Train Predictions')
plt.plot(range(train_data_length, train_data_length + len(test_predict)), test_predict, label='Test Predictions')

plt.title(f"{stock_symbol} Stock Price Prediction Using LSTM")
plt.xlabel("Date")
plt.ylabel("Close Price USD")
plt.legend()
plt.show()
```
### Output:
![image](https://github.com/user-attachments/assets/cee4e979-f2de-420c-94c8-358c475fae59)


### Result:
Thus the system was trained successfully and the prediction was carried out.
