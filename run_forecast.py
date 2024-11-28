import multiprocessing
from multiprocessing import Pool
from functools import partial
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from forecaster import Forecaster
import numpy as np

class RunForecast(Forecaster):
    """
    A class that extends to run forecasting methods in parallel using multiprocessing.
    
    This class runs all specified forecasting methods in parallel and computes evaluation metrics
    when performing backtesting.
    
    Args:
        y (pd.DataFrame): Endogenous variable(s) (target time series).
        X (pd.DataFrame): Exogenous variables (features).
        methods (list): List of forecasting methods to run.
        start_date (pd.Timestamp, optional): Start date of the data.
        end_date (pd.Timestamp, optional): End date of the data.
        horizons (list, optional): List of forecast horizons to evaluate.
        runtype (str, optional): Type of run ('Tuning', 'Backtest', 'Forecasting').
        scale (str, optional): Scaling method ('log' or None).
        alpha (float, optional): Regularization parameter for Lasso.
    """
    def __init__(self, y, X, methods, start_date=None, end_date=None,
                 horizons=[1, 3, 6, 12], runtype='Backtest', scale='log', alpha=0.00005):
        super().__init__(y=y, X=X, horizon=None, training_period=None, series=None,
                         runtype=runtype, scale=scale, start_date=start_date, end_date=end_date,
                         methods=methods, alpha=alpha)
        self.methods = methods
        self.horizons = horizons
        self.results = {}
        self.metrics = {}
    
    def run_forecasting(self, series, horizon):
        """
        Runs all specified forecasting methods for a single series and horizon.
        
        Args:
            series (str): The target time series column name.
            horizon (int): The forecast horizon.
        
        Returns:
            dict: A dictionary containing forecasts and metrics for each method.
        """
        # Update horizon and training period
        self.horizon = horizon
        self.training_period = horizon  # Assuming training period equals horizon
        self.series = series
        
        # Recompute date range and forecast index
        self.date_range = self.getDR()
        self.forecastindex = self.forecastindex()
        
        # Prepare a dictionary to store forecasts and metrics
        forecasts = {}
        metrics = {}
        
        # Create an instance for the current series and horizon
        instance = Forecaster(
            y=self.y, X=self.X, horizon=self.horizon, training_period=self.training_period,
            series=self.series, runtype=self.runtype, scale=self.scale,
            start_date=self.startdate, end_date=self.enddate, alpha=self.alpha
        )
        
        # Initialize methods mapping
        methods_mapping = {
            'ARIMA': instance.ARIMA,
            'FBProphet': instance.FB_Prophet,
            'HW': instance.HW,
            'Theta': instance.Theta,
            'VAR': instance.VAR_preds,
            'XGBoost': instance.XGBoost,
            'RNN': instance.RNN,
            'NBEATS': instance.NBEATS
        }
        
        # Run each method and collect forecasts
        for method in self.methods:
            if method in methods_mapping:
                try:
                    if method == 'VAR':
                        # VAR_preds returns multiple outputs
                        _, _, forecast = methods_mapping[method]()
                    elif method == 'XGBoost':
                        # XGBoost requires varmod as input
                        varmod = instance.VAR_preds()[1]  # Get VAR model data
                        forecast, _ = methods_mapping[method](varmod)
                    else:
                        forecast = methods_mapping[method]()
                    
                    forecasts[method] = forecast
                    
                    # If backtesting, compute metrics
                    if self.runtype == 'Backtest':
                        y_true = self.yModel.loc[self.forecastindex, self.series]
                        y_pred = forecast
                        metrics[method] = self.compute_metrics(y_true, y_pred)
                except Exception as e:
                    print(f"Error running method {method} for series {series}: {e}")
            else:
                print(f"Method {method} is not recognized.")
        
        return {'forecasts': forecasts, 'metrics': metrics}
    
    def compute_metrics(self, y_true, y_pred):
        """
        Computes evaluation metrics between true values and predictions.
        
        Args:
            y_true (pd.Series): Actual values.
            y_pred (pd.Series or np.ndarray): Predicted values.
        
        Returns:
            dict: A dictionary containing RMSE, MAE, and MAPE.
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        rmse_value = np.sqrt(mean_squared_error(y_true, y_pred))
        mae_value = mean_absolute_error(y_true, y_pred)
        mape_value = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return {'RMSE': rmse_value, 'MAE': mae_value, 'MAPE': mape_value}
    
    def run_all(self, series_list):
        """
        Runs forecasting for all series and horizons using multiprocessing.
        
        Args:
            series_list (list): List of target series to forecast.
        
        Updates:
            self.results: A dictionary containing forecasts for each series and horizon.
            self.metrics: A dictionary containing metrics for each series and horizon.
        """
        # Create a list of tasks (series, horizon combinations)
        tasks = [(series, horizon) for series in series_list for horizon in self.horizons]
        
        # Use multiprocessing Pool to run tasks in parallel
        with Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.starmap(self.run_forecasting, tasks)
        
        # Organize results
        for idx, (series, horizon) in enumerate(tasks):
            key = f"{series}_horizon_{horizon}"
            self.results[key] = results[idx]['forecasts']
            self.metrics[key] = results[idx]['metrics']
    
    def get_results(self):
        """
        Returns the forecasting results.
        
        Returns:
            dict: Dictionary containing forecasts for each series and horizon.
        """
        return self.results
    
    def get_metrics(self):
        """
        Returns the computed metrics.
        
        Returns:
            dict: Dictionary containing metrics for each series and horizon.
        """
        return self.metrics