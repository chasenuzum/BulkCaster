BulkCaster

BulkCaster is a comprehensive time series forecasting tool designed to handle multiple time series data efficiently. It provides a streamlined workflow for preprocessing data, performing forecasts using various models, and visualizing the results. BulkCaster includes a user-friendly Streamlit app for interactive forecasting and supports multiprocessing for handling large datasets.

Table of Contents

	•	Features
	•	Folder Structure
	•	Installation
	•	Usage
	•	Running the Streamlit App
	•	Running Forecasts via Script
	•	Forecasting Methods
	•	Examples
	•	Dependencies
	•	Contributing
	•	License

Features

	•	Multiple Forecasting Models: Supports a variety of forecasting methods including ARIMA, Facebook Prophet, Holt-Winters, Theta, VAR, XGBoost, and deep learning models from Darts.
	•	Streamlit App: Interactive web application for uploading data, selecting forecasting methods, and visualizing forecasts.
	•	Batch Processing: Ability to run forecasts on multiple time series simultaneously.
	•	Multiprocessing Support: Efficient processing of large datasets using multiprocessing.
	•	Customizable: Easy to extend with additional forecasting models or preprocessing steps.
	•	Evaluation Metrics: Computes performance metrics such as RMSE, MAE, and MAPE during backtesting.

Folder Structure

bulkcaster/
├── data/
│   └── sample_data.csv
├── app.py
├── forecaster.py
├── pricing_tools.py
├── readme.md
├── requirements.txt
└── run_forecast.py

	•	data/: Directory containing sample datasets or user data.
	•	app.py: Streamlit application for interactive forecasting.
	•	forecaster.py: Core module containing the BulkCaster class and forecasting methods.
	•	pricing_tools.py: (Optional) Module for pricing-related utilities (if applicable).
	•	readme.md: Documentation and usage instructions.
	•	requirements.txt: List of required Python packages.
	•	run_forecast.py: Script for running forecasts outside of the Streamlit app.

Installation

	1.	Clone the repository:

git clone https://github.com/yourusername/bulkcaster.git
cd bulkcaster


	2.	Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate


	3.	Install required packages:

pip install -r requirements.txt

Ensure that you have the latest versions of pip and setuptools:

pip install --upgrade pip setuptools



Usage

Running the Streamlit App

The Streamlit app provides an interactive interface for uploading your time series data, selecting forecasting methods, and visualizing the forecasts.
	1.	Start the Streamlit app:

streamlit run app.py


	2.	Upload Data:
	•	Prepare your time series data in a CSV file.
	•	Ensure the CSV file contains a timestamp column and one or more numerical columns representing different time series.
	•	The timestamp column should be in a datetime format or convertible to datetime.
	3.	Interact with the App:
	•	Select the timestamp column after uploading your data.
	•	Choose the forecasting methods you wish to use.
	•	Set the forecast horizon and training data range.
	•	Run the forecast and view the results.

Running Forecasts via Script

You can run forecasts directly using the run_forecast.py script, which utilizes the BulkCaster class.
	1.	Prepare your data:
	•	Place your CSV file in the data/ directory or specify the path in the script.
	•	Ensure your data is properly formatted with a timestamp index.
	2.	Configure the script:
	•	Open run_forecast.py and set the desired parameters, such as:
	•	Path to the data file.
	•	List of time series columns to forecast.
	•	Forecasting methods to use.
	•	Forecast horizon.
	3.	Run the script:

python run_forecast.py


	4.	View Results:
	•	The script will output the forecasts and evaluation metrics.
	•	Results can be saved to files or printed to the console.

Forecasting Methods

BulkCaster supports the following forecasting methods:
	•	ARIMA: Autoregressive Integrated Moving Average model.
	•	Facebook Prophet: Time series forecasting model by Facebook.
	•	Holt-Winters: Exponential smoothing with trend and seasonality.
	•	Theta Method: Decomposition-based forecasting method.
	•	VAR: Vector Autoregression for multivariate time series.
	•	XGBoost: Extreme Gradient Boosting with support for VAR inputs.
	•	Darts RNN: Recurrent Neural Network models from the Darts library.
	•	Darts N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting.

Examples

Example: Forecasting Using the Streamlit App

	1.	Launch the app:

streamlit run app.py


	2.	Upload your data file (e.g., sample_data.csv).
	3.	Select the timestamp column and ensure your data looks correct.
	4.	Choose the forecasting methods and set the forecast horizon.
	5.	Click “Run Forecast” to generate forecasts.
	6.	Visualize the forecasts and download results if needed.

Example: Running Forecasts via Script

	1.	Edit run_forecast.py to specify your data file and parameters.

# run_forecast.py

import pandas as pd
from forecaster import BulkCaster

# Load your data
data = pd.read_csv('data/sample_data.csv', parse_dates=['Date'], index_col='Date')

# Initialize the BulkCaster
bulkcaster = BulkCaster(
    y=data[['Series1', 'Series2']],
    X=data.drop(columns=['Series1', 'Series2']),
    methods=['ARIMA', 'XGBoostVARND'],
    horizons=[12],
    runtype='Backtest'
)

# Run forecasts
bulkcaster.run_all(series_list=['Series1', 'Series2'])

# Retrieve and display results
forecasts = bulkcaster.get_results()
metrics = bulkcaster.get_metrics()
print(forecasts)
print(metrics)


	2.	Run the script:

python run_forecast.py



Dependencies

Ensure that the following Python packages are installed (as specified in requirements.txt):
	•	pandas
	•	numpy
	•	statsmodels
	•	scikit-learn
	•	xgboost
	•	pmdarima
	•	prophet
	•	darts
	•	streamlit
	•	matplotlib
	•	plotly

Install all dependencies using:

pip install -r requirements.txt

Note: Some packages may have additional dependencies or require system libraries (e.g., Prophet may require pystan or cmdstanpy). Refer to the package documentation for installation details.

Contributing

Contributions are welcome! Please follow these steps:
	1.	Fork the repository.
	2.	Create a new branch for your feature or bugfix:

git checkout -b feature/your-feature-name


	3.	Commit your changes with descriptive messages.
	4.	Push your branch to your forked repository.
	5.	Submit a pull request to the main repository.

Please ensure your code adheres to best practices and includes appropriate documentation.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Disclaimer: This README provides an overview of the BulkCaster tool based on the files and functionalities specified. For detailed information and the latest updates, please refer to the actual code and documentation within the repository.