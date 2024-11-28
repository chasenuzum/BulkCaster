# app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO
from forecaster import Forecaster
from run_forecast import RunForecast 

# Set page configuration
st.set_page_config(
    page_title="Time Series Forecasting App",
    page_icon="📈",
    layout="wide",
)

# Title of the app
st.title("Time Series Forecasting App")

# Instructions
st.write("""
Upload a CSV file containing time series data. The file should have a timestamp column and one or more columns of numerical data representing different time series.

**Instructions:**
- Ensure your data has a timestamp column.
- Each column represents a different time series.
- The timestamp column should be in a datetime format or convertible to datetime.
""")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Data preview:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

    # Ensure the DataFrame has a timestamp column
    timestamp_col = None
    potential_timestamp_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if potential_timestamp_cols:
        timestamp_col = st.selectbox("Select the timestamp column:", options=potential_timestamp_cols)
    else:
        st.error("No timestamp column found. Please ensure your CSV file has a timestamp column.")
        st.stop()

    # Convert the timestamp column to datetime
    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    except Exception as e:
        st.error(f"Error converting timestamp column to datetime: {e}")
        st.stop()

    # Set the timestamp column as the index
    df.set_index(timestamp_col, inplace=True)

    # Drop any non-numeric columns (excluding the timestamp)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found for forecasting.")
        st.stop()

    # Allow user to select forecasting methods
    default_methods = ['ARIMA', 'FBProphet', 'HW', 'Theta', 'VAR', 'XGBoost', 'RNN', 'NBEATS']
    selected_methods = st.multiselect(
        "Select forecasting methods to use:",
        options=default_methods,
        default=default_methods
    )

    # Forecast horizon input
    horizon = st.number_input(
        "Select the forecast horizon (number of periods to forecast):",
        min_value=1,
        max_value=60,
        value=12,
        step=1
    )

    # Start and end date for training data
    start_date = st.date_input(
        "Select the start date for training data:",
        value=df.index.min().date(),
        min_value=df.index.min().date(),
        max_value=df.index.max().date()
    )
    end_date = st.date_input(
        "Select the end date for training data:",
        value=df.index.max().date(),
        min_value=df.index.min().date(),
        max_value=df.index.max().date()
    )

    # Scale option
    scale_option = st.selectbox(
        "Select scaling method:",
        options=['log', 'none'],
        index=0
    )
    scale = 'log' if scale_option == 'log' else None

    # Forecast button
    if st.button("Run Forecast"):
        # Placeholder for results
        forecast_results = {}
        feature_importances = {}
        metrics = {}

        # Iterate over each time series column
        for series in numeric_cols:
            st.write(f"Forecasting series: {series}")

            # Prepare the data for the Valcast class
            y = df[[series]]
            X = df.drop(columns=[series])  # Use other series as exogenous variables

            # Initialize the Valcast instance
            valcast_instance = Valcast(
                y=y,
                X=X,
                horizon=horizon,
                training_period=horizon,  # You can adjust this as needed
                series=series,
                runtype='Forecast',
                scale=scale,
                start_date=pd.to_datetime(start_date),
                end_date=pd.to_datetime(end_date),
                methods=selected_methods
            )

            # Dictionary to hold forecasts for the current series
            series_forecasts = {}

            # Run each selected forecasting method
            for method in selected_methods:
                try:
                    if method == 'ARIMA':
                        forecast = valcast_instance.ARIMA()
                    elif method == 'FBProphet':
                        forecast = valcast_instance.FB_Prophet()
                    elif method == 'HW':
                        forecast = valcast_instance.HW()
                    elif method == 'Theta':
                        forecast = valcast_instance.Theta()
                    elif method == 'VAR':
                        _, _, forecast = valcast_instance.VAR_preds()
                        forecast = forecast[series]
                    elif method == 'XGBoost':
                        # need VAR model data
                        _, varmodND, _ = valcast_instance.VAR_preds()
                        forecast, feature_importance = valcast_instance.XGBoostVARND(varmodND)
                        feature_importances[f"{series}_{method}"] = feature_importance
                    elif method == 'RNN':
                        forecast = valcast_instance.Darts_RNN()
                    elif method == 'NBEATS':
                        forecast = valcast_instance.Darts_NBEATS()
                    else:
                        st.warning(f"Method {method} is not recognized.")
                        continue

                    # Store the forecast
                    series_forecasts[method] = forecast

                    # Optionally, compute metrics if actual values are available
                    # For now, we'll skip this as we're forecasting into the future

                except Exception as e:
                    st.error(f"Error forecasting series {series} with method {method}: {e}")
                    continue

            # Store forecasts for the current series
            forecast_results[series] = series_forecasts

        # Display the forecasts
        for series, forecasts in forecast_results.items():
            st.subheader(f"Forecasts for {series}")
            for method, forecast in forecasts.items():
                st.write(f"**{method}**")
                if isinstance(forecast, pd.Series) or isinstance(forecast, pd.DataFrame):
                    st.line_chart(forecast)
                else:
                    # Convert to DataFrame if it's not already
                    forecast_df = pd.DataFrame(forecast, index=valcast_instance.forecastindex, columns=[series])
                    st.line_chart(forecast_df)

        # Display feature importances if available
        if feature_importances:
            st.subheader("Feature Importances from XGBoost")
            for key, importance_df in feature_importances.items():
                st.write(f"**{key}**")
                st.dataframe(importance_df)