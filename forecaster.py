import datetime
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
import itertools
import xgboost as xgb
import pickle as pkl
import operator
from sklearn.linear_model import Lasso
from statsmodels.tsa.api import VAR
from prophet import Prophet
from pmdarima import auto_arima, ARIMA
from sktime.forecasting.theta import ThetaForecaster
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error as mse
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool, freeze_support, Lock
from itertools import islice
from sklearn.base import clone
# Darts imports
from darts import TimeSeries
from darts.models import (
    RNNModel,
    NBEATSModel,
)
from darts.utils.timeseries_generation import datetime_attribute_timeseries

warnings.simplefilter(action='ignore', category=FutureWarning)


class Forecaster:
    """
    Forecasting tool that produces tuned, optimized forecasts using various models.

    Args:
        y (pd.DataFrame): Endogenous variable(s) (target time series).
        X (pd.DataFrame): Exogenous variables (features).
        horizon (int): Forecast horizon.
        training_period (int): Length of training period.
        series (str): Target time series column name.
        runtype (str): Type of run ('Tuning' or 'Forecasting').
        scale (str, optional): Scaling method ('log' or None). Defaults to 'log'.
        start_date (pd.Timestamp, optional): Start date of the data. Defaults to '2012-01-01'.
        end_date (pd.Timestamp, optional): End date of the data. Defaults to today's date.
        methods (list, optional): List of forecasting methods.
        smooth_factor (float, optional): Smoothing factor. Defaults to None.
        alpha (float, optional): Regularization parameter for Lasso. Defaults to 0.00005.
    """

    def __init__(self, y, X, horizon, training_period, series, runtype, scale='log',
                 start_date=pd.to_datetime('2012-01-01'), end_date=pd.to_datetime('today'),
                 methods=None, smooth_factor=None, alpha=0.00005):
        self.today = pd.to_datetime('today')
        self.startdate = start_date
        self.enddate = end_date
        self.horizon = horizon
        self.runtype = runtype
        self.methodnames = methods or ['ARIMA', 'FBProphet', 'HW', 'VAR', 'XGBoostND',
                                       'Theta', 'RNN', 'NBEATS']
        self.series = series
        self.alpha = alpha
        self.tunedate = self.today - pd.offsets.MonthBegin(horizon + 2)

        self.date_range = self.getDR()
        self.scale = scale
        self.smooth = smooth_factor

        self.y = y  # Endogenous variables
        self.X = X  # Exogenous variables
        self.training_period = training_period

        # Cleaned and prepared endogenous variable
        self.yclean, self.ylog = self.yPrep()
        self.yModel = self.ylog if self.scale == 'log' else self.yclean

        # Cleaned and prepared exogenous variables
        self.Xclean = self.jPrep()
        self.forecastindex = self.forecastindex()
        self.jqual = self.qualdata()

        # Lasso data
        self.lasdata = self.lasso_data()
        self.lock = Lock()

        if runtype == 'Tuning':
            # Variable selection and tuning
            self.VARTune()

    # Data Processing Methods
    def dediff(self, todaysVal, forecast):
        """
        Reverts differenced time series data back to its original scale.

        Args:
            todaysVal (float): The last known value before forecasting.
            forecast (array-like): Differenced forecasted values.

        Returns:
            array-like: Forecasted values in the original scale.
        """
        future = forecast.copy()
        for i in range(len(forecast)):
            if i == 0:
                future[i] = todaysVal + forecast[0]
            else:
                future[i] = future[i - 1] + forecast[i]
        return future

    def jPrep(self, difference=False):
        """
        Prepares exogenous variables/features for modeling.

        Args:
            difference (bool, optional): Whether to difference the data. Defaults to False.

        Returns:
            pd.DataFrame: Cleaned exogenous variables ready for modeling.
        """
        scale = MinMaxScaler()
        j = pd.DataFrame(data=scale.fit_transform(self.X), index=self.X.index, columns=self.X.columns)
        if difference:
            j = j.diff().replace(0, np.nan).fillna(method='ffill').fillna(method='bfill')
        else:
            j = j.replace(0, np.nan).fillna(method='ffill').fillna(method='bfill')
        j.columns = j.columns.str.replace(r"[^\w\s]", "").str.replace(" ", "_")
        return j

    def yPrep(self):
        """
        Prepares endogenous variables/features for modeling.

        Returns:
            tuple: Cleaned dataframes (yclean, ylog).
        """
        yclean = pd.DataFrame(data=self.y, index=self.y.index, columns=self.y.columns)
        yclean = yclean.replace(0, np.nan).fillna(method='ffill').fillna(method='bfill')
        yclean.columns = yclean.columns.str.replace(r"[^\w\s]", "_").str.replace(" ", "_")

        if self.smooth is not None:
            yclean = yclean.ewm(alpha=self.smooth, axis=0).mean()

        ylog = np.log(yclean.copy())

        ylog = ylog.asfreq('MS')
        yclean = yclean.asfreq('MS')

        return yclean, ylog

    def clean_data(self, difference=False):
        """
        Cleans and joins exogenous and endogenous data.

        Args:
            difference (bool, optional): Whether to difference the data. Defaults to False.

        Returns:
            pd.DataFrame: Cleaned data ready for modeling.
        """
        yclean = self.yModel.copy()
        j = self.Xclean.copy()
        data = yclean.join(j)
        return data

    def forecastindex(self, freq='MS'):
        """
        Generates the forecast index based on the horizon.

        Args:
            freq (str, optional): Frequency of the time series. Defaults to 'MS'.

        Returns:
            pd.DatetimeIndex: Index for forecasting periods.
        """
        fhindex = pd.date_range(start=self.enddate + pd.offsets.MonthBegin(1), freq=freq, periods=self.horizon)
        return fhindex

    def qualdata(self):
        """
        Prepares qualitative data for modeling (e.g., month dummies).

        Returns:
            pd.DataFrame: Dataframe containing qualitative features.
        """
        qualdata = self.Xclean.copy(deep=True)
        qualdata['Date'] = qualdata.index
        jqual = qualdata[['Date']]

        jqual['forward'] = jqual['Date'] + pd.offsets.MonthBegin(self.horizon)
        jqual = pd.DataFrame(
            data=pd.date_range(self.date_range[0], jqual['forward'][-1], freq='MS').tolist(),
            index=pd.date_range(self.date_range[0], jqual['forward'][-1], freq='MS'),
            columns=['Date']
        )
        jqualmonths = pd.get_dummies(jqual['Date'].dt.month, drop_first=True)
        return jqualmonths

    def autodata(self):
        """
        Generates autoregressive features.

        Returns:
            tuple: Dataframes containing autoregressive features (yautoND, yautoD).
        """
        ydauto = self.yModel.replace(0, np.nan).copy()
        ydauto = ydauto.fillna(method='ffill').fillna(method='bfill')

        yautoND = ydauto.copy()
        ydauto = pd.DataFrame(ydauto.diff().dropna())

        autocols = [
            ydauto.columns + '_Auto1', ydauto.columns + '_Auto3',
            ydauto.columns + '_Auto6', ydauto.columns + '_Auto9',
            ydauto.columns + '_Auto12'
        ]
        flat_cols = [item for sublist in autocols for item in sublist]

        yautoD = pd.concat([
            ydauto.shift(1), ydauto.shift(3), ydauto.shift(6),
            ydauto.shift(9), ydauto.shift(12)
        ], axis=1, ignore_index=True)
        yautoD.columns = flat_cols

        yautoND = pd.concat([
            yautoND.shift(1), yautoND.shift(3), yautoND.shift(6),
            yautoND.shift(9), yautoND.shift(12)
        ], axis=1, ignore_index=True)
        yautoND.columns = flat_cols
        return yautoND, yautoD

    def getDR(self):
        """
        Generates the date range for training.

        Returns:
            pd.DatetimeIndex: Date range.
        """
        startdate = self.startdate.date()
        enddate = (self.enddate - pd.offsets.MonthBegin(0)).date()
        return pd.date_range(startdate, enddate, freq='MS')

    # Univariate Forecasting Methods
    def ARIMA(self):
        """
        Performs ARIMA forecasting.

        Returns:
            pd.Series: Forecasted values.
        """
        arimadata = self.yModel.loc[self.date_range, self.series].copy()
        arimadata = arimadata.replace([np.inf, -np.inf], np.nan)
        arimadata = arimadata.fillna(method='ffill').fillna(method='bfill')
        try:
            arima_mod = auto_arima(arimadata, m=12, start_p=0, start_q=0, suppress_warnings=True, maxiter=50)
        except ValueError:
            arima_mod = ARIMA([0, 0, 0]).fit(arimadata)

        try:
            arima_fore = arima_mod.predict(n_periods=self.horizon)
        except ValueError:
            arima_fore = pd.Series(data=np.nan, index=self.forecastindex, name=self.series)
            arima_fore[self.series] = arimadata[-1]

        if self.scale == 'log':
            return np.exp(arima_fore)
        else:
            return arima_fore

    def FB_Prophet(self):
        """
        Performs forecasting using Facebook Prophet.

        Returns:
            pd.Series: Forecasted values.
        """
        df = pd.DataFrame(self.yModel[[self.series]], columns=[self.series])
        df = df.loc[self.date_range]
        df.insert(loc=0, column='Date', value=df.index)
        df = df.rename(columns={'Date': 'ds', self.series: 'y'})

        prophet = Prophet(seasonality_mode='additive', mcmc_samples=0, daily_seasonality=False, weekly_seasonality=False)
        prophet.fit(df)
        futureframe = prophet.make_future_dataframe(periods=self.horizon, include_history=False, freq='MS')
        forecast = prophet.predict(futureframe)
        if self.scale == 'log':
            return np.exp(forecast['yhat'])
        else:
            return forecast['yhat']

    def Theta(self):
        """
        Performs forecasting using the Theta method.

        Returns:
            pd.Series: Forecasted values.
        """
        df = pd.DataFrame(self.yModel[[self.series]], columns=[self.series])
        df = df.loc[self.date_range]
        df.insert(loc=0, column='Date', value=df.index)
        df['DateSK'] = df['Date'].dt.to_period('M')
        thetaTrain = pd.Series(data=df[self.series], name=self.series)
        thetaTrain.index = df['DateSK']
        try:
            theta_model = ThetaForecaster(sp=12).fit(thetaTrain)
        except ValueError:
            theta_model = ThetaForecaster(sp=12, deseasonalize=False).fit(thetaTrain)
        ypred = theta_model.predict(fh=[x for x in range(1, self.horizon + 1)])
        if self.scale == 'log':
            return np.exp(ypred)
        else:
            return ypred

    def HW(self):
        """
        Performs Holt-Winters exponential smoothing forecasting.

        Returns:
            pd.Series: Forecasted values.
        """
        try:
            hw = sm.tsa.ExponentialSmoothing(
                self.yModel.loc[self.date_range, self.series],
                trend='additive',
                seasonal='multiplicative',
                seasonal_periods=12
            ).fit()
        except ValueError:
            hw = sm.tsa.ExponentialSmoothing(self.yModel.loc[self.date_range, self.series]).fit()

        hwforecast = hw.predict(start=self.forecastindex[0], end=self.forecastindex[-1])
        if self.scale == 'log':
            return np.exp(hwforecast)
        else:
            return hwforecast

    # Tuning Methods
    def lasso_data(self):
        """
        Prepares data for Lasso regression.

        Returns:
            pd.DataFrame: Dataframe suitable for Lasso regression.
        """
        ylas = self.yModel.copy()
        ydiff = ylas.diff().iloc[1:]
        jdiff = self.Xclean.diff().iloc[1:, :]
        lasdat = ydiff.join(jdiff)
        lasdat = lasdat.fillna(method="ffill").fillna(method="bfill")
        return lasdat

    def make_endvals(self):
        """
        Retrieves the last known values for endogenous and exogenous variables.

        Returns:
            pd.DataFrame: Dataframe containing the last known values.
        """
        yModel = self.yModel
        X = self.Xclean
        endvals = yModel.join(X)
        dates = self.date_range
        dates = pd.to_datetime(dates)
        endvals = endvals.loc[dates]
        return endvals.iloc[[-1]]

    def VARTune(self):
        """
        Tunes a VAR (Vector Autoregression) model by selecting the best parameters based on RMSE.

        This method performs feature selection using Lasso regression and hyperparameter tuning
        over the number of predictors and lags for the VAR model. It iterates over combinations
        of predictors (`num_predictors`) and lags (`num_lags`), selects features using Lasso
        regression adjusted by `alpha`, fits a VAR model, forecasts, and computes the RMSE.
        The combination with the lowest RMSE is selected and stored in `self.var_tuned_params`.

        Steps:
        1. Generate combinations of the number of predictors and lags.
        2. For each combination:
            a. Adjust `alpha` in Lasso regression to select the desired number of features.
            b. Fit a VAR model with the selected features and specified lags.
            c. Forecast using the VAR model.
            d. Compute RMSE between the forecast and test data.
        3. Store the best parameters based on the lowest RMSE.

        Attributes:
            self.var_tuned_params (dict): Dictionary containing the best parameters (`num_predictors`, `num_lags`, `selected_features`).
        """
        # Define the ranges for the number of predictors and lags
        num_predictors_list = list(range(3, 7))  # Possible numbers of predictors to select
        num_lags_list = list(range(3, 10))       # Possible numbers of lags to use in the VAR model

        # Prepare the data
        lasso_data = self.lasso_data()
        y_test = self.yModel.loc[self.forecastindex, self.series]
        y_train = lasso_data.loc[self.date_range[1:], self.series]
        X_train = lasso_data.loc[self.date_range[1:], self.Xclean.columns.tolist()]

        # Initialize a list to store the RMSE scores and corresponding parameters
        scores = []
        starting_alpha = self.alpha

        # Iterate over all combinations of number of predictors and lags
        for num_predictors, num_lags in itertools.product(num_predictors_list, num_lags_list):
            alpha = starting_alpha
            count = 0

            # Adjust alpha to select the desired number of features using Lasso regression
            while True:
                # Perform Lasso regression for feature selection
                lasso = Lasso(alpha=alpha, fit_intercept=True)
                lasso.fit(X_train, y_train)

                # Get coefficients and select features with non-zero coefficients
                coefficients = pd.Series(lasso.coef_, index=X_train.columns)
                selected_features = coefficients[coefficients.abs() > 0].index.tolist()

                # Break the loop if the desired number of features is achieved
                if len(selected_features) <= num_predictors:
                    break
                else:
                    # Increment alpha to reduce the number of selected features
                    if count <= 200:
                        alpha += starting_alpha * 0.01
                    elif 200 < count <= 1000:
                        alpha += starting_alpha * 0.1
                    else:
                        alpha += starting_alpha * 0.5
                    count += 1

            # Prepare training data with the selected features
            VAR_train = X_train[selected_features].join(y_train)

            # Fit the VAR model with the specified number of lags
            model = VAR(endog=VAR_train)
            try:
                model_results = model.fit(maxlags=num_lags)
            except Exception as e:
                print(f"VAR model fitting failed for num_predictors={num_predictors}, num_lags={num_lags}: {e}")
                continue  # Skip this combination if fitting fails

            # Forecast future values
            try:
                forecast = model_results.forecast(y=VAR_train.values[-model_results.k_ar:], steps=self.horizon)
            except Exception as e:
                print(f"VAR forecasting failed for num_predictors={num_predictors}, num_lags={num_lags}: {e}")
                continue  # Skip if forecasting fails

            # Revert differencing if applicable
            endvals_for = self.make_endvals()[VAR_train.columns]
            forecast = self.dediff(endvals_for, forecast)

            # Create a DataFrame for the forecast
            forecast_index = pd.date_range(self.forecastindex[0], periods=self.horizon, freq='MS')
            forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=VAR_train.columns)

            # Compute RMSE between the forecast and actual values
            rmse_value = self.rmse(y_true=y_test, y_pred=forecast_df[self.series])

            # Store the results
            score = {
                'rmse': rmse_value,
                'params': {'num_predictors': num_predictors, 'num_lags': num_lags},
                'selected_features': selected_features
            }
            scores.append(score)

        # Select the best parameters based on the lowest RMSE
        if scores:
            self.var_tuned_params = min(scores, key=lambda x: x['rmse'])
        else:
            print("No valid VAR model configurations were found.")
            self.var_tuned_params = None

    def TuningMin(self, model, param_grid, X_train, y_train, X_test, y_test):
        """
        Performs hyperparameter tuning using grid search and returns the best parameters.

        This method iterates over all combinations of hyperparameters specified in `param_grid`,
        fits the model on the training data, evaluates it on the test data, and selects the
        hyperparameters that result in the lowest RMSE.

        Args:
            model (estimator): The machine learning model to be tuned.
            param_grid (dict): Dictionary specifying the hyperparameters and their possible values.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing target.

        Returns:
            dict: A dictionary containing the best hyperparameters and corresponding RMSE.
                Format: {'rmse': float, 'params': dict}
        """
        # Initialize a list to store the scores
        scores = []

        # Iterate over all combinations of hyperparameters
        for params in ParameterGrid(param_grid):
            # Clone the model to avoid modifying the original
            test_model = clone(model).set_params(**params)

            try:
                # Fit the model with early stopping
                test_model.fit(
                    X_train, y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    early_stopping_rounds=25,
                    verbose=False
                )

                # Predict on the test data using the best iteration
                y_pred = test_model.predict(X_test, iteration_range=(0, test_model.best_ntree_limit))

                # Calculate RMSE
                rmse_value = self.rmse(y_true=y_test, y_pred=y_pred)

                # Store the result
                score = {'rmse': rmse_value, 'params': params}
                scores.append(score)
            except Exception as e:
                print(f"Error with parameters {params}: {e}")
                continue  # Skip to the next parameter set if an error occurs

        # Return the parameters with the minimum RMSE
        if scores:
            best_score = min(scores, key=lambda x: x['rmse'])
            return best_score
        else:
            print("No valid hyperparameter configurations were found.")
            return None

    def VAR_preds(self):
        """
        Generates forecasts using the VAR model with tuned parameters.

        This method uses the parameters obtained from `VARTune` to fit a VAR model and generate forecasts.
        It prepares the training data with the selected features, fits the VAR model, and forecasts future
        values. The method returns the forecasted values in both differenced and non-differenced forms, and
        applies an exponential transformation if the data was log-scaled.

        Returns:
            tuple:
                - forecast_df_diff (pd.DataFrame): Forecasted values in differenced form.
                - forecast_df (pd.DataFrame): Forecasted values in non-differenced form.
                - forecast_final (pd.DataFrame): Final forecasted values, transformed if necessary.
        """
        # Check if tuning has been performed
        if not hasattr(self, 'var_tuned_params') or self.var_tuned_params is None:
            print("VAR model parameters not tuned. Please run VARTune() before forecasting.")
            return None, None, None

        # Retrieve the best parameters from VARTune
        params = self.var_tuned_params

        # Prepare the data
        lasso_data = self.lasso_data()
        X_train = lasso_data.loc[self.date_range[1:], self.Xclean.columns.tolist()]
        y_train = lasso_data[self.series].loc[self.date_range[1:]]

        # Prepare training data with selected features
        VAR_train = X_train[params['selected_features']].join(y_train)

        # Fit the VAR model with the selected number of lags
        model = VAR(endog=VAR_train)
        try:
            model_results = model.fit(maxlags=params['params']['num_lags'])
        except Exception as e:
            print(f"VAR model fitting failed: {e}")
            return None, None, None

        # Forecast future values
        try:
            forecast = model_results.forecast(y=VAR_train.values[-model_results.k_ar:], steps=self.horizon)
        except Exception as e:
            print(f"VAR forecasting failed: {e}")
            return None, None, None

        # Revert differencing to obtain non-differenced forecast
        endvals_for = self.make_endvals()[VAR_train.columns]
        forecast_nd = self.dediff(endvals_for, forecast.copy())

        # Create DataFrames for the forecasts
        forecast_index = pd.date_range(self.forecastindex[0], periods=self.horizon, freq='MS')
        forecast_df_diff = pd.DataFrame(forecast, index=forecast_index, columns=VAR_train.columns)
        forecast_df = pd.DataFrame(forecast_nd, index=forecast_index, columns=VAR_train.columns)

        # Apply exponential transformation if data was log-scaled
        if self.scale == 'log':
            forecast_final = forecast_df.apply(np.exp)
        else:
            forecast_final = forecast_df

        # Return the forecasts
        return forecast_df_diff, forecast_df, forecast_final

    def NBEATS(self):
        """
        Performs forecasting using the N-BEATS model from Darts.

        Returns:
            pd.Series: Predictions.
        """
        series = self.prepare_darts_series()
        model = NBEATSModel(
            input_chunk_length=self.training_period,
            output_chunk_length=self.horizon,
            n_epochs=300,
            random_state=42
        )
        model.fit(series)
        forecast = model.predict(n=self.horizon)
        if self.scale == 'log':
            return np.exp(forecast.values())
        else:
            return forecast.values()

    def RNN(self):
        """
        Performs forecasting using a Recurrent Neural Network model from Darts.

        Returns:
            pd.Series: Predictions.
        """
        series = self.prepare_darts_series()
        model = RNNModel(
            model='LSTM',
            input_chunk_length=self.training_period,
            output_chunk_length=self.horizon,
            n_epochs=300,
            random_state=42,
            likelihood=None
        )
        model.fit(series)
        forecast = model.predict(n=self.horizon)
        if self.scale == 'log':
            return np.exp(forecast.values())
        else:
            return forecast.values()

    def prepare_darts_series(self):
        """
        Prepares the time series data for Darts models.

        Returns:
            TimeSeries: Darts TimeSeries object.
        """
        y_series = self.yModel[self.series].copy()
        y_series = y_series.fillna(method='ffill').fillna(method='bfill')
        y_series.index = pd.to_datetime(y_series.index)
        series = TimeSeries.from_series(y_series)
        return series

    # Multivariate Forecasting Methods (XGBoost models)
    def XGBoost(self, varmod, tuning=True):
        """
            Performs forecasting using XGBoost on non-differenced data.

            This method trains an XGBoost model using non-differenced data, incorporating autoregressive features
            and exogenous variables. It supports hyperparameter tuning and returns both the predictions and
            feature importance scores.

            Args:
                varmod (pd.DataFrame): DataFrame containing exogenous variables and the target variable from the VAR model.
                tuning (bool, optional): Whether to perform hyperparameter tuning. Defaults to True.

            Returns:
                tuple:
                    - y_pred (np.ndarray): Predicted values for the forecast horizon.
                    - feature_importance_df (pd.DataFrame): DataFrame containing feature importance scores.
        """
        # Prepare the target variable
        # Fill missing values and select the target series
        y_model = self.yModel.fillna(method="ffill").fillna(method="bfill")
        y_model = y_model[[self.series]]

        # Combine endogenous and exogenous variables for training
        # Join the target variable with cleaned exogenous variables
        train_data = self.yModel.join(self.Xclean, how='left')

        # Prepare test target values and define hyperparameter grid if tuning
        if self.runtype in ['Backtest', 'Tuning']:
            # Actual target values for the forecast period
            y_test = self.yModel.loc[self.forecastindex, self.series]

            # Hyperparameter grid for tuning
            param_grid = {
                'objective': ["reg:squarederror"],
                'max_depth': [4, 6, 8, 12],
                'n_estimators': [100, 250, 2500],
                'learning_rate': [0.05, 0.1, 0.25, 0.5],
                'min_child_weight': [1, 3],
                'seed': [34]
            }

        # Set random seed for reproducibility
        np.random.seed(34)

        # Prepare exogenous variables for testing (X_test), excluding the target variable
        # Exclude the last column (assumed to be the target variable) from varmod
        X_test = varmod.iloc[:, :-1]

        # Get the columns used in varmod
        varmod_columns = varmod.columns

        # Prepare the training data with columns matching varmod
        # Select columns in training data that correspond to varmod columns
        train_data = train_data[varmod_columns]
        # Use data up to the last date in the date range
        train_data = train_data.loc[:self.date_range[-1]]

        # Separate endogenous and exogenous variables in training data
        y_train = train_data.iloc[:, -1]  # Target variable (last column)
        X_train = train_data.iloc[:, :-1]  # Exogenous variables

        # Prepare autoregressive features and add them to training and testing data
        if self.horizon <= 12:
            # Obtain non-differenced autoregressive features
            y_auto_nd = self.autodata()[0]
            # Select autoregressive features for the target variable
            auto_cols = [col for col in y_auto_nd.columns if col.startswith(str(varmod_columns[-1]))]
            y_autos = y_auto_nd[auto_cols]

            # Map horizon to the number of lags to include
            horizon_lag_mapping = {12: 1, 9: 2, 6: 3, 3: 4, 1: len(auto_cols)}
            n_lags = horizon_lag_mapping.get(self.horizon, 1)

            # Select the last n_lags columns from y_autos
            selected_autos = y_autos.iloc[:, -n_lags:]

            # Add autoregressive features to training and testing exogenous variables
            X_train = X_train.join(selected_autos)
            X_test = X_test.join(selected_autos)

        # Add a random variable 'Rand' to detect overfitting or assess variable importance
        X_train['Rand'] = np.random.normal(size=len(X_train))
        X_test['Rand'] = np.random.normal(size=len(X_test))

        # Add qualitative data (e.g., month dummies) to exogenous variables
        X_train = X_train.join(self.jqual)
        X_test = X_test.join(self.jqual)


        # Train the XGBoost model
        if self.runtype == 'Tuning':
            # Perform hyperparameter tuning
            tuning_results = self.TuningMin(
                model=xgb.XGBRegressor(),
                paramsNOCV=param_grid,
                Xtrain=X_train,
                ytrain=y_train,
                Xtest=X_test,
                ytest=y_test
            )
            best_params = tuning_results['params']

            # Initialize and train the model with the best parameters
            xgb_model = xgb.XGBRegressor(**best_params)
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                early_stopping_rounds=25,
                eval_metric='rmse',
                verbose=False
            )
            # Predict using the best number of trees determined during training
            y_pred = xgb_model.predict(X_test, iteration_range=(0, xgb_model.best_ntree_limit))
        else:
            # Train the model with default parameters
            xgb_model = xgb.XGBRegressor()
            xgb_model.fit(X_train, y_train)
            # Predict using the trained model
            y_pred = xgb_model.predict(X_test)

        # Compute feature importance
        importance_type = 'gain'  # Type of feature importance metric
        feature_importance = xgb_model.get_booster().get_score(importance_type=importance_type)
        # Sort features by importance in descending order
        importance_sorted = sorted(feature_importance.items(), key=operator.itemgetter(1), reverse=True)
        # Create a DataFrame for feature importance
        feature_importance_df = pd.DataFrame(importance_sorted, columns=['feature', 'importance'])

        # Step 15: Apply exponential transformation if data was log-scaled
        if self.scale == 'log':
            y_pred = np.exp(y_pred)

        # Step 16: Return the predictions and feature importance DataFrame
        return y_pred, feature_importance_df

    # Accuracy Measures
    def MAPE(self, y_true, y_pred):
        """
        Calculates Mean Absolute Percentage Error.

        Args:
            y_true: Actual values.
            y_pred: Predicted values.

        Returns:
            float: MAPE value.
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return (np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    def rmse(self, y_true, y_pred):
        """
        Calculates Root Mean Squared Error.

        Args:
            y_true: Actual values.
            y_pred: Predicted values.

        Returns:
            float: RMSE value.
        """
        return sqrt(mse(y_true, y_pred))

    def MPE(self, y_true, y_pred):
        """
        Calculates Mean Percentage Error.

        Args:
            y_true: Actual values.
            y_pred: Predicted values.

        Returns:
            float: MPE value.
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return (np.mean((y_true - y_pred) / y_true) * 100).round(2)