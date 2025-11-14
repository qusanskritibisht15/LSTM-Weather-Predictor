# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Step 1: Load and preprocess the data
# Replace 'weather_data.csv' with your dataset path
data = pd.read_csv('weather_data.csv')

# Visualize the data
plt.plot(data['Temperature'], label='Temperature')
plt.title("Temperature Trend")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.legend()
plt.show()

# Normalize the data (use all columns as features)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Temperature', 'Humidity', 'Wind Speed', 'Pressure']])

# Create sequences for the LSTM model
sequence_length = 305  # Use the last 60 readings to predict the next one
X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, :])  # Use all features
    y.append(scaled_data[i, 0])  # Predict temperature

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))  # Reshape for LSTM input

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)  # Output layer (predicting temperature)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Step 3: Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Step 4: Evaluate the model
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Step 5: Make predictions
predicted_temperatures = model.predict(X_test)
predicted_temperatures = scaler.inverse_transform(np.hstack((predicted_temperatures, np.zeros((predicted_temperatures.shape[0], 3)))))[:, 0]

# Plot actual vs predicted temperatures
plt.plot(scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 3))))), label="Actual Temperatures")
plt.plot(predicted_temperatures, label="Predicted Temperatures")
plt.title("Actual vs Predicted Temperatures")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.legend()
plt.show()

# Step 6: Forecast future values
# Use the last available data for forecasting
last_sequence = scaled_data[-sequence_length:]
last_sequence = last_sequence.reshape((1, sequence_length, last_sequence.shape[1]))
forecasted_value = model.predict(last_sequence)
forecasted_value = scaler.inverse_transform(np.hstack((forecasted_value, np.zeros((forecasted_value.shape[0], 3)))))

print("Forecasted Temperature for the next time step:", forecasted_value[0][0])

