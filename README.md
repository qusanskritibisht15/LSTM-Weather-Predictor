# LSTM-Weather-Predictor

**Summary**
This project forecasts temperature using historical weather data with an Long Short-Term Memory LSTM-based deep learning model.The dataset for time-series weather forecasting includes temperature, humidity, wind speed, and pressure. After normalization and training, the model predicts future temperature trends and visualizes actual temperature; forecasted results showcasing neural network-based time series prediction.

**Technical Overview**

Model Type: Sequential LSTM Neural Network

Objective: Predict future temperature trends

Frameworks: TensorFlow / Keras

Dataset: weather_data.csv containing multiple weather parameters

**Libraries Used**

* NumPy
* Pandas
* Scikit-learn
* TensorFlow / Keras
* Matplotlib

**Approach**

**Data Preprocessing**

Loaded and visualized the weather dataset using Pandas and Matplotlib.
Normalized key features (Temperature, Humidity, Wind Speed, Pressure) using MinMaxScaler.
Transformed the dataset into a sequential format suitable for LSTM-based time-series forecasting.

**Model Development**

Designed a two-layer LSTM model with Dropout layers to reduce overfitting.
Added Dense layers for final output prediction.
Compiled the model using the Adam optimizer and Mean Squared Error (MSE) as the loss function.

**Model Training and Evaluation**

Trained the model for 20 epochs with an 80:20 train-test split.
Evaluated model performance using training and validation loss curves.
Achieved a low validation loss (~0.0091), reflecting good predictive accuracy.

**Prediction and Forecasting**

Compared actual and predicted temperature values through visual analysis.
Used the most recent data sequence to forecast future temperature values.

**Results**

The LSTM model effectively captured temporal patterns in the dataset.
Forecasted next temperature: 26.56°C
Training and validation loss curves showed stable learning and strong generalization.

**Visualizations**

Temperature trend over time
Training vs Validation loss curve
Actual vs Predicted temperature comparison

**Conclusion**

This project demonstrates how LSTM-based deep learning models can accurately forecast weather trends using historical data. By identifying temporal dependencies, the model delivers reliable temperature predictions and can be extended to other parameters such as humidity, wind speed, or rainfall.

**Author**

Sanskriti Bisht

Data Science Trainee
