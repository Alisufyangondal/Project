# Sunspot Number Forecasting Project

This project focuses on forecasting sunspot numbers using various time series analysis techniques. The goal is to predict future sunspot activity based on historical data.

## Notebook: forecast_V6.ipynb

### Overview

The `forecast_V7.ipynb` notebook explores several time series models to forecast sunspot numbers. It includes data preprocessing, model building, training, and evaluation. The notebook uses historical sunspot data to predict future trends.

### Libraries Used

*   pandas
*   numpy
*   matplotlib
*   seaborn
*   tensorflow
*   sklearn
*   statsmodels
*   arch
*   prophet

### Models Implemented

1.  **LSTM (Long Short-Term Memory) Model:**
    *   A neural network model used for sequence prediction.
    *   Evaluated using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared.

2.  **Prophet Model:**
    *   A time series forecasting model developed by Facebook.
    *   Evaluated using MSE, MAE, RMSE, Mean Absolute Scaled Error (MASE), R-squared, and Explained Variance Score.

3.  **ARIMA (Autoregressive Integrated Moving Average) Model:**
    *   A classical time series model.
    *   Evaluated using MAE, MSE, RMSE, R-squared, and MASE.

4.  **SARIMA (Seasonal ARIMA) Model:**
    *   An extension of ARIMA to handle seasonal data.
    *   Evaluated using MAE, MSE, RMSE, R-squared, and Explained Variance Score.

5.  **GARCH (Generalized Autoregressive Conditional Heteroskedasticity) Model:**
    *   A model for time series volatility.
    *   Evaluated using MAE, MSE, RMSE, R-squared, MASE, and Explained Variance Score.

6.  **Holt-Winters Exponential Smoothing:**
    *   A method for forecasting time series data with seasonality.
    *   Evaluated using MAE, MSE, RMSE, R-squared, and Explained Variance Score.

### Key Steps

1.  **Data Loading and Preprocessing:**
    *   Loading the sunspot dataset.
    *   Creating a date column.
    *   Scaling the data using MinMaxScaler.

2.  **Data Splitting:**
    *   Splitting the data into training and testing sets.

3.  **Model Building and Training:**
    *   Building and training various time series models.

4.  **Prediction and Evaluation:**
    *   Making predictions on the test data.
    *   Evaluating the models using appropriate metrics.

5.  **Visualization:**
    *   Plotting actual vs. predicted values.
    *   Visualizing training and validation loss.
    *   Plotting residuals.

### Results

The notebook provides a comprehensive analysis of sunspot number forecasting using different models. Evaluation metrics are calculated for each model to assess their performance. Plots and visualizations are used to interpret the results and compare the models.

### Conclusion

This project demonstrates the application of various time series analysis techniques for forecasting sunspot numbers. The results and insights gained from this project can be valuable for understanding and predicting future sunspot activity.
