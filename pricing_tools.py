"""
Pricing Module for Forecasting Future Prices of Quantity Demanded
"""
from typing import Union
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from math import sqrt
from sklearn.model_selection import train_test_split
import pickle as pkl
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from sdv.single_table import GaussianCopulaSynthesizer as CGAN
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
from torch import set_float32_matmul_precision as set_pre
from xgboost import DMatrix

set_pre('medium')

class PricingTools:
    'General tools/methods for handling pricing-related data'

    def __init__(self) -> None:
        pass

    def save_obj(self, obj, name):
        modelpik = name +  '.pkl'
        savemod = open(modelpik, 'wb')
        pkl.dump(obj, savemod,  pkl.HIGHEST_PROTOCOL)
        savemod.close()

    def extract_features_information(self, X: pd.DataFrame):
        '''Extracts and categorizes features in the dataset.

        Args:
        X (pd.DataFrame): Input DataFrame containing features.

        Returns:
        tuple: Tuple containing continuous variables, qualitative variables, and dates.
        '''
        contVarL = []
        datesVar = []
        qualVars = []

        for i in X.columns:
            if X[i].dtype in [np.float64, np.int64]:
                contVarL.append(i)
            elif X[i].dtype == np.dtype('datetime64[ns]'):
                datesVar.append(i)
            elif X[i].dtype == np.object:
                qualVars.append(X[i].astype('str'))

        contVar = X[contVarL] if contVarL else None
        qual = pd.concat(qualVars, axis=1) if qualVars else None
        date = X[datesVar[0]] if len(datesVar) == 1 else X[datesVar] if len(datesVar) > 1 else None

        return contVar, qual, date

    def create_ml_features(self, cont: list, qual: list, onehotmodel, standardmodel):
        '''Creates machine learning features from provided data.

        Args:
        cont (pd.DataFrame): DataFrame of continuous variables.
        qual (pd.DataFrame): DataFrame of qualitative variables.
        onehotmodel: One-hot encoder model.
        standard: Standard scaler model.

        Returns:
        pd.DataFrame: DataFrame of machine learning features.
        '''
        if cont is None:
            # Handle empty 'cont' DataFrame
            return pd.DataFrame()

        xqual = onehotmodel.transform(qual) if qual is not None else pd.DataFrame()
        xqual = pd.DataFrame(data=xqual, columns=onehotmodel.get_feature_names_out())

        xcont = standardmodel.transform(cont)
        xcont = pd.DataFrame(data=xcont, columns=standardmodel.get_feature_names_out())

        return pd.concat([xqual, xcont], axis=1)

    def train_test_split(self, modeldata, price):
        '''Splits data into training and testing sets.

        Args:
        modeldata: Data for model training.
        price: Pricing data.

        Returns:
        tuple: Tuple containing training and testing datasets for features and price.
        '''
        X_train, X_test, y_train, y_test = train_test_split(
            modeldata, price, train_size=.75, random_state=34
        )
        return X_train, X_test, y_train, y_test
    
    def transform_X(self, X):
        """
        Transform and encode input features for machine learning prediction.

        Parameters:
            X (DataFrame, Series, or dict): The input data containing features to be transformed.

        Returns:
            DataFrame: The transformed and encoded input data as a pandas DataFrame.
        """
        # Convert X to a DataFrame if it is a Series or a dict
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        elif isinstance(X, dict):
            X = pd.DataFrame(X, index=[0])
        elif isinstance(X, pd.DataFrame):
            X = X
        
        # Make a Dmatrix
        XDmax = DMatrix(X)
        return XDmax
    
class SyntheticData(PricingTools):
    '''Creates synthetic data with SDV package.'''

    def __init__(self, premodel_data, cont, qual, onehot_model, standard_scaler,
                 future_price_level=None, invoice_date_name=None):
        super().__init__()
        self.cont = cont
        self.qual = qual
        self.premodel_data = premodel_data
        self.future_price_level = future_price_level
        self.top_corr = self.future_price_level.columns.tolist()[1]
        self.cont.append(self.top_corr)
        self.invoice_date_name = invoice_date_name
        self.onehot_model = onehot_model
        self.standard_scaler = standard_scaler

    def synthetic_monte(self, invoice_forecast: Union[pd.Series, pd.DataFrame]):
        '''
        Generates synthetic data using Monte Carlo simulations based on the provided invoice forecast.

        Parameters:
        - invoice_forecast (pd.DataFrame): DataFrame containing forecasted invoice data.

        Returns:
        - pd.DataFrame: Synthetic data generated from Monte Carlo simulations.
        '''
        print('Training synthetic Monte Carlo model')

        # Set up metadata and distributions
        self.forecast = invoice_forecast
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(self.premodel_data)

        for column in self.qual:
            metadata.update_column(column_name=column, sdtype='categorical')

        # keep for later testing of distributions
        # numerical_distributions = {column: 'beta' for column in self.cont.columns.tolist()}

        self.sampler = CGAN(metadata, enforce_min_max_values=False, enforce_rounding=False)
        # future setting we toy with later... don't delete!
        # verbose = True, cuda = True, numerical_distributions=numerical_distributions)
        # epochs = 100, discriminator_dim = (128, 128), generator_dim = (128,128), embedding_dim = 128, batch_size=100_000)
        self.sampler.fit(self.premodel_data)

        # Prepare future sales dates from forecasted invoices
        date_list = [[i] * int(j) for i, j in zip(invoice_forecast['Date'], invoice_forecast['value'])]
        cache_flatdates = [item for sublist in date_list for item in sublist]
        self.dates = cache_flatdates
        # Merge future price levels with forecast
        future = pd.merge(left=invoice_forecast, right=self.future_price_level,
                          left_on='Date', right_on='index', how='left')
        future[self.top_corr] = future[self.top_corr].fillna(method='ffill').fillna(method='bfill')
        # print(future.iloc[:, 1])
        future['Month'] = 'Month_' + pd.to_datetime(future.loc[:, 'Date']).dt.month.astype('str')
        conditions = []
        print(future)
        for i in future.index:
            condition = Condition(num_rows=int(future.iloc[i]['value']),
                                  column_values={self.top_corr: float(future.iloc[i][self.top_corr]),
                                                 'Month': str(future.iloc[i]['Month'])})
            conditions.append(condition)

        synthetic_data = self.sampler.sample_from_conditions(conditions=conditions, max_tries_per_batch=50_000)
        # print(synthetic_data.columns.tolist())

        # Merge price levels with synthetic data
        self.monte_carlo_data = pd.concat([synthetic_data, pd.Series(cache_flatdates, name='future_dates')], axis=1)

        self.future_X = super().create_ml_features(cont=synthetic_data.loc[:, self.cont],
                                           qual=synthetic_data.loc[:, self.qual],
                                           onehotmodel=self.onehot_model,
                                           standardmodel=self.standard_scaler)\
                                           .fillna(method='ffill').fillna(method='bfill')

        return print('Finished Synthetic Data Generation!')

    
class CollectPricingData(PricingTools):
    '''A class to manage pricing-related data collection and processing.

    Args:
        y: The pricing data.
        X: The feature dataset.
        horizon: The forecasting horizon.
        inflation_matrix: Matrix containing inflation data.
        invoice_dates: Dates of the invoices.
        scale (str): Scaling method (default is 'linear').
        monthly_unit: Unit for monthly pricing.
        price_level (str): Level of pricing (default is 'point').
        end_date (pd.Timestamp): End date for data collection (default is today).
        segment (str): Segment for pricing (default is 'Irrigation').
    '''
    def __init__(
        self,
        y,
        X,
        horizon,
        inflation_matrix,
        invoice_dates,
        scale='linear',
        monthly_unit=None,
        price_level='Point',
        end_date=pd.to_datetime('today'),
        segment='Irrigation'
    ):
        super().__init__()
        self.scale = scale
        self.end_date = pd.to_datetime(str(end_date.year) + '-' + str(end_date.month) + '-' + '01')
        self.y = y
        self.ylog = self.y.apply(np.log)
        if scale == 'log':
            self.price = self.ylog
        else:
            self.price = self.y
        self.segment = segment
        self.inDates = pd.to_datetime(invoice_dates)
        self.datemonthbound = self.inDates - pd.offsets.MonthBegin(1)
        self.horizon = horizon
        self.price_forecast_level = price_level
        self.monthlyunit = monthly_unit
        if self.monthlyunit is None:
            self.monthlyunit = self.y
        else:
            self.monthlyunit  = self.monthlyunit         
        self.X = X
        self.target = y.name
        self.invoice_date_name = invoice_dates.name
        self.l3years = self.shorten_df()
        self.price_levels = inflation_matrix
        self.clean_price_levels()
        self.calculate_correlation()
        self.topCorr = self.correlationDF.iloc[1,].name
        self.corr_forecast = self.forecast_levels()
        self.future_price_levels_df = self.future_price_levels()
        self.premodeldata = self.prepare_premodel_data()
        self.cont, self.qual, self.date = super().extract_features_information(self.premodeldata)
        self.premodeldata = pd.concat([self.cont, self.qual, self.date], axis=1)
        self.onehotmodel = OneHotEncoder(handle_unknown='infrequent_if_exist', sparse=False, min_frequency=.0015).fit(self.qual)
        self.standardmodel = StandardScaler().fit(self.cont)
        self.modeldata = super().create_ml_features(self.cont,
                                                    self.qual,
                                                    onehotmodel=self.onehotmodel,
                                                    standardmodel=self.standardmodel)
        self.Xtrain, self.Xtest, self.ytrain, self.ytest = super().train_test_split(modeldata=self.modeldata,
                                                                                    price=self.price)


    def clean_price_levels(self):
        self.price_levels.columns = self.price_levels.columns.str.replace(".", "").str.replace(",", "").str.replace("-", "").str. \
            replace("$", "").str.replace("=", "").str.replace("/", "").str.replace("'", "").str.replace("(", "").str. \
            replace(")", "").str.replace("&", "").str.replace("10", "Ten").str. \
            replace("____", "_").str.replace("___", "_").str.replace("__", "_").str.replace(":", "").str. \
            replace("321", "Three_Two_One_").str. \
            replace("+", "").str.replace(">", "").str.replace("[", "").str.replace("]", "").str. \
            replace('"', "").str.replace("%", "").str.replace("#", "").str.replace("  ", "_").str.replace(" ", "_").str.replace("___", "_").str.replace("__", "_").str.replace("__", "_")
        self.price_levels.index = pd.to_datetime(self.price_levels.index) - pd.offsets.MonthBegin(1)
        self.price_levels.index = self.price_levels.index + pd.offsets.MonthBegin(2)

    
    def shorten_df(self):
        '''Shortens the DataFrame to the last 3 years based on the provided invoice dates.

        Returns:
        pd.DataFrame: DataFrame containing data from the last 3 years.
        '''
        l3df = pd.concat([self.X, self.inDates], axis=1)
        l3df = l3df.loc[l3df[self.inDates.name].dt.year >= pd.to_datetime('today').year - 3]
        return l3df
    
    def calculate_monthly_prices(self):
        '''Creates monthly pricing data based on the invoice date.

        Returns:
        pd.DataFrame: DataFrame containing monthly pricing data.
        '''
        cross_data = pd.concat([self.monthlyunit, self.inDates], axis=1)

        if self.segment == 'UtilityNA':
            monthly_data = cross_data.groupby([cross_data[self.invoice_date_name].dt.year, cross_data[self.invoice_date_name].dt.month]).median()
        else:
            monthly_data = cross_data.groupby([cross_data[self.invoice_date_name].dt.year, cross_data[self.invoice_date_name].dt.month]).mean()

        monthly_data['Year'] = monthly_data.index.get_level_values(0)
        monthly_data['Month'] = monthly_data.index.get_level_values(1)
        monthly_data['Day'] = 1
        monthly_data.index = pd.to_datetime(dict(year=monthly_data['Year'], month=monthly_data['Month'], day=monthly_data['Day']))

        return monthly_data

    
    def calculate_correlation(self):
        '''Calculates correlation between variables and monthly average pricing.

        Returns:
        pd.DataFrame: DataFrame containing correlation values.
        '''
        monthlydataEx = self.price_levels.join(self.calculate_monthly_prices())
        if self.segment == 'IrrigationNA':
            monthlydataEx = monthlydataEx['2011-01-01':]
        else:
            monthlydataEx = monthlydataEx['2014-01-01':]
            monthlydataEx = monthlydataEx.rolling(2).mean()['2015-01-01':pd.to_datetime('today') - pd.offsets.MonthBegin(6)]

        correlationEx = monthlydataEx.corr()
        self.correlationDF = correlationEx.loc[:, [self.monthlyunit.name]].sort_values(self.monthlyunit.name, ascending=False)
        # print(self.correlationDF)


    def forecast_levels(self):
        '''Generates forecasted price levels using a time series model.

        Returns:
        pd.DataFrame: DataFrame containing forecasted price levels.
        '''
        h = self.horizon
        
        # Retrieve price levels data
        price_levels_df = self.price_levels.copy()
        
        # Create inflation index and clean data
        price_levels_df['Citi_Surprise'] = price_levels_df.Citi_Inflation_Surprise_Index_United_States.rolling(3).mean().fillna(method='ffill').fillna(method='bfill')
        price_levels_df = price_levels_df.fillna(method='ffill')
        ts_data = TimeSeries.from_dataframe(price_levels_df.loc['2008-01-01':, :], freq='MS')
        start_date = pd.to_datetime('2011-12-01')
        
        # Get top correlation
        target_level = self.topCorr
        targetP = ts_data[target_level]
        targetP = targetP.drop_before(start_date)

        inflation = ts_data['Citi_Surprise'].shift(h)
        inflation = inflation.drop_before(start_date)
        
        # Darts scaling for DL time series model
        price_transformer = Scaler()
        price_transformed = price_transformer.fit_transform(targetP)
        
        # Create month and year covariate series
        start_date_forecast = pd.to_datetime('2012-01-01')
        periods_cov = len(targetP) + h
        
        # Create year series with Darts package
        year_series = datetime_attribute_timeseries(
            pd.date_range(start=start_date_forecast, freq=targetP.freq_str, periods=periods_cov),
            attribute="year",
            one_hot=False
        )
        
        # Seasonal Features
        year_series = Scaler().fit_transform(year_series)
        month_series = datetime_attribute_timeseries(
            year_series, attribute="month", one_hot=True
        )
        
        # Scale all features, including inflation / Citi index and stack on year TS
        inflation_scale = Scaler().fit_transform(inflation)
        covariates = year_series.stack(month_series)
        covariates = covariates.stack(inflation_scale)
        quantiles = [.05, .25, .5, .75, .95]

        # Model initialization in darts
        price_model = RNNModel(
            model="LSTM",
            hidden_dim=14,
            dropout=0,
            batch_size=12,
            n_epochs=150,
            optimizer_kwargs={"lr": 1e-3},
            random_state=42,
            training_length=8,
            input_chunk_length=8,
            likelihood=QuantileRegression(quantiles=quantiles)
        )
        price_model.fit(price_transformed, future_covariates=covariates, verbose=True)
        
        # Sampling for probabilistic price
        y_pred_P = price_model.predict(h, num_samples=5000)
        y_pred_P_unscale = price_transformer.inverse_transform(y_pred_P)
        
        # Price index forecast with three different outcomes: high, low, point
        y_forecast_P, y_min_P, y_max_P = (
            y_pred_P_unscale.quantile(.5).pd_dataframe(),
            y_pred_P_unscale.quantile(.05).pd_dataframe(),
            y_pred_P_unscale.quantile(.95).pd_dataframe()
        )
        index_forecasts_P = pd.concat([y_forecast_P, y_min_P, y_max_P], axis=1)
        index_forecasts_P.columns = [f"{target_level}_Point", f"{target_level}_Low", f"{target_level}_High"]
        
        return index_forecasts_P

    def future_price_levels(self):
        '''Generates future price levels based on historical and forecasted data.

        Returns:
        pd.Series: Series containing future price levels.
        '''
        # Retrieve the last three observations of historical prices
        historical_price = self.price_levels.iloc[-3:].loc[:, self.topCorr].fillna(method='ffill').reset_index(drop=True)

        # Retrieve forecasted price index
        price_index_forecast = self.corr_forecast.loc[:, f"{self.topCorr}_{self.price_forecast_level}"].reset_index(drop=True)

        # Combine historical and forecasted price index
        combined_forecast = historical_price.tolist() + price_index_forecast.tolist()

        # Create a Pandas Series with the combined data
        price_forecast_series = pd.Series(data=combined_forecast, name=self.topCorr)

        # Set the index for the forecasted prices
        price_forecast_series.index = pd.date_range(start=self.end_date, freq='MS', periods=len(price_forecast_series))

        # Fill missing values using forward fill and backward fill methods
        price_forecast_series = price_forecast_series.fillna(method='ffill').fillna(method='bfill')
        price_forecast_series.name = self.topCorr

        return price_forecast_series

    def prepare_premodel_data(self):
        '''
        Prepares premodel data by joining and merging necessary dataframes.

        Returns:
        pd.DataFrame: Premodel data after joining and merging.
        '''
        premodel_data = self.X.join(self.datemonthbound).merge(
            self.price_levels.loc[:, self.topCorr],
            how='left',
            left_on=self.invoice_date_name,
            right_on=self.price_levels.index
        )
        return premodel_data
    

class PriceForecast(PricingTools):
    '''Class to forecast and summarize future prices.'''

    def __init__(self, regressor, future_X, scale='linear', dates=None):
        '''
        Initializes PriceForecast object.

        Args:
        - regressor: The regressor used for prediction.
        - future_X: DataFrame containing future predictor variables.
        - scale: Scale type ('linear' or 'log').
        - date_col: Column name for date in future_X.
        '''
        super().__init__()
        self.regressor = regressor
        self.future_X = future_X
        self.future_dates = dates
        self.scale = scale

    def run_pricing(self):
        '''
        Runs the pricing model and generates a summary forecast.

        Returns:
        - str: Completion message.
        '''
        if self.regressor is None:
            return 'Please pass a regressor to price future orders...'
        else:
            self.predict_price()
            self.summary_forecast()
            return 'Pricing Complete!'

    def predict_price(self):
        '''
        Predicts future prices based on the regressor.

        Returns:
        - pd.Series: Series containing predicted future prices.
        '''
        self.future_y = self.regressor.predict(self.transform_X(self.future_X))

        if self.scale == 'log':
            self.future_y = np.exp(self.future_y)

        self.future_prices_Series = pd.Series(data=self.future_y, name='Price', index=self.future_dates)
        
    def summary_forecast(self):
        '''
        Generates a summary forecast of future prices.

        Returns:
        - pd.DataFrame: DataFrame summarizing forecasted prices.
        '''
        revenue_forecast = pd.DataFrame({'Price': self.future_y, 'Date': self.future_dates})
        start_date = self.future_dates[0]

        # Total forecast
        revenue_month = revenue_forecast.groupby([revenue_forecast.Date.dt.year, revenue_forecast.Date.dt.month]).sum()
        horizon = revenue_month.shape[0]

        # Price Profile
        profiles = revenue_forecast.groupby([revenue_forecast.Date.dt.year, revenue_forecast.Date.dt.month]).median()
        quantity = revenue_forecast.groupby([revenue_forecast.Date.dt.year, revenue_forecast.Date.dt.month]).count()

        revenue_month.columns = revenue_month.columns + '_Sums'
        profiles.columns = profiles.columns + '_Profile'
        quantity.columns = quantity.columns + '_Quantity'

        future_state = pd.concat([revenue_month, profiles, quantity], axis=1)
        future_state.index = pd.date_range(start_date, periods=horizon, freq='MS')
        self.future_state = future_state