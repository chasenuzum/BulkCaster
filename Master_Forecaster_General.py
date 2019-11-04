# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 16:14:27 2019

Create script that runs all time series through 
HW, FB, and XGBoost

@author: chasenuzum
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import statsmodels.api as sm
from fbprophet import Prophet
import operator
import matplotlib.pyplot as plt


#---Select forecast length, up to 12---
h = 6

#---Select date range---
date_range = pd.date_range('2012-01-01', '2019-01-01', freq='MS') 

#---Select Macros, MEIs---
#Forecast all time series; forecast out
macdataloc = open(r'') #locate MEIs file path
j = pd.read_csv(macdataloc)

#Macros on date; cleaned in R
j['Date'] = j['Date'].astype('datetime64[ns]')
j.index = (j['Date'])
j = j.replace(0, np.nan)
j = j.fillna(j.mean())

#---Select endogenous data---
endodataloc = open(r'') #locate clean endogenous file path
y = pd.read_csv(endodataloc)

#Endo on date; clean here
y['Date'] = y['Date'].astype('datetime64[ns]')
y.index = (y['Date'])
#y = y.drop(columns = [])
y1 = y.loc[date_range]
#Replace 0 with nans
y1 = y1.replace(0, np.nan)

y1 = y1.fillna(y1.mean()) #mean to replace missing values
ylog = np.log(y1.iloc[:, 1:]) #can remove log if needed
ylog['ds'] = y['Date']

#---Columns to loop; called 'teams'---
teams = []

"""
#---OPTIONAL: Simple moving average of y for smoothing out training, can blank this if you like---
smoothy = []
for ave in ylog.iloc[:, :-1]:
    movingave = ylog[ave].shift().rolling(6,min_periods=1).mean()
    smoothy.append(movingave)

smoothy = pd.concat(smoothy, axis=1)
smoothy['ds'] = date_range
ylog = smoothy.dropna()

#Check simple moving average for smoothness
plt.plot(ylog.index,ylog[''])
plt.show()
"""

#---Prophet---
fb_all = []
for team in teams:
    subdf = ylog[['ds', team]]
    subdf = ylog.rename(columns={'ds':'ds', team:'y'})
    fb = Prophet(seasonality_mode = 'additive', mcmc_samples = 150) #trying new seasonality
    fb.fit(subdf)
    fbmodel = fb.make_future_dataframe(periods=h, include_history=False, freq ='MS')
    fbforecast = fb.predict(fbmodel)
    fb_all.append(fbforecast[['ds','yhat']])
    
fb_forecasts = pd.concat(fb_all, axis=1)
fb_forecasts = fb_forecasts.set_index(fbforecast['ds'], inplace=False)
fb_forecasts = fb_forecasts.drop(columns = ['ds'])
fb_forecasts.columns = teams
fb_forecasts = np.exp(fb_forecasts)

#---XGBoost---
#Clean for XGBoost
j['proxyDate'] = (j['Date'] - pd.offsets.MonthBegin(-h)) #look back h months
j.index = j['proxyDate'] #set index h months back
j = j.drop(columns = [ 'Date', 'proxyDate']) #drop unused columns
#Training data for teams 
trainendoxgb = ylog[teams] #endo = teams

#Training data for MEIs
trainexoxgb = trainendoxgb.join(j, how='left') # line up the dates to teams
trainexoxgb = trainexoxgb.drop(columns =teams) # drop teams from training x's

#Dates that represent forecast length; facebook for proxy, not forecasting, gets us close
actualendoxgb = pd.DataFrame(np.log(fb_forecasts))

#Matches up to actual values of forecast periods for macros
actualexoxgb = j.loc[actualendoxgb.index].copy()


#Model w/ xgboost
xgb_all = []
feature_list = []
for i in teams:
    ytrain, Xtrain = trainendoxgb[[i]], trainexoxgb
    ytest, Xtest = actualendoxgb[[i]], actualexoxgb
    dtrain = xgb.DMatrix(Xtrain,label=ytrain)
    dtest = xgb.DMatrix(Xtest,label=ytest) #Use xgb matrices for efficiency
    xgboostreg = xgb.XGBRegressor(n_estimators=1000) #using base parameters

    xgboostreg.fit(Xtrain, ytrain,
               eval_set=[(Xtrain, ytrain), (Xtest, ytest)],
               early_stopping_rounds=50,
               verbose=False)
    ypred = xgboostreg.predict(Xtest)         
    xgb_all.append(ypred)
    f='gain'
    featureImportance = xgboostreg.get_booster().get_score(importance_type= f)
    topFeatures = {k: v for k, v in featureImportance.items() if v > 0.5}

    importance = sorted(featureImportance.items(), key=operator.itemgetter(1), reverse = True)
    fscore = pd.DataFrame(importance, columns=['feature', i])
    feature_list.append(fscore)
xgbforecasts = pd.DataFrame.from_items(zip(teams, xgb_all))
xgbforecasts = xgbforecasts.set_index(fbforecast['ds'], inplace=False)
xgbforecasts = xgbforecasts.astype('float64')
xgbforecasts = np.exp(xgbforecasts)

features = pd.concat(feature_list, axis=1)

#---Holt-Winters---
hw_all = []
ylog.index.freq = 'MS'
for z in teams:
    hw = sm.tsa.ExponentialSmoothing(ylog[[z]], trend='additive', seasonal='additive',
                                     seasonal_periods = 12).fit()
    hwforecast = hw.predict(start = xgbforecasts.index[0],end=xgbforecasts.index[-1])
    hw_all.append(hwforecast)
    
hw_forecasts = pd.concat(hw_all, axis=1)
hw_forecasts = hw_forecasts.set_index(fbforecast['ds'], inplace=False)
hw_forecasts.columns = teams
hw_forecasts = np.exp(hw_forecasts)

#To csvs for excel analysis
fb_forecasts.to_csv('FB_Forecasts.csv', index=True)
xgbforecasts.to_csv('XGB_Forecasts.csv', index=True)
hw_forecasts.to_csv('HW_Forecasts.csv', index=True)
features.to_csv("Best_MEIs.csv", index=False)

#---BLANK THIS SECTION TO END IF LOOKING FORWARD/NOT USING ACTUAL DATA TO TEST---

#Check MAPE
#Calculate the MAPE for our methods above
yact = y.loc[actualendoxgb.index].copy()

def mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return (np.mean(np.abs((y_true - y_pred) / y_true)) * 100).round(2)

fbmape = []
xbgmape = []
hwmape = []
for z in teams:
    fbmapes = mape(yact[[z]], fb_forecasts[[z]])
    fbmape.append(fbmapes)
for q in teams:
    xgboostmape = mape(yact[[q]], xgbforecasts[[q]])
    xbgmape.append(xgboostmape)
for u in teams:
    hwmapes = mape(yact[[u]], hw_forecasts[[u]])
    hwmape.append(hwmapes)
    
fbmape = pd.DataFrame(fbmape).T
fbmape.columns = teams
xbgmape = pd.DataFrame(xbgmape).T
xbgmape.columns = teams
hwmape = pd.DataFrame(hwmape).T
hwmape.columns = teams
mapes = pd.concat([fbmape, xbgmape, hwmape], axis = 0)
mapes.insert(0, "Method", ['FB_Prophet', 'XGBoost', 'HW'], True)
mapes.to_csv('MAPE.csv', index=False)

#Plotting
from bokeh.plotting import figure
from bokeh.models import Title
from bokeh.io import export_png

for w in teams:
    xgbplot = figure(title = 'XGBoost Forecast -' + w, plot_width=800, plot_height=600, 
                     x_axis_type='datetime')
    xgbplot.line(y1.index.values, y1[w], 
                 color = 'red', line_width=3, alpha=0.5, 
                 legend='Training')
    xgbplot.line(xgbforecasts.index.values, xgbforecasts[w], 
                 color = 'blue', line_width=3, line_dash='dashed',
                 alpha=0.5, legend='Forecast')
    xgbplot.line(yact.index.values, yact[w], 
                 color = 'blue', line_width=3,
                 alpha=0.5, legend='Actual')
    xgbplot.legend.location = 'top_left'
    xgbplot.add_layout(Title(text="MAPE: "+ mape(yact[[w]], xgbforecasts[[w]]).astype(str), align="center"), "below")

    export_png(xgbplot, filename='XGB_Forecast_' + w + '.png')
    
for o in teams:
    hwplot = figure(title = 'Holt-Winters Forecast -' + o, plot_width=800, plot_height=600, 
                     x_axis_type='datetime')
    hwplot.line(y1.index.values, y1[o], 
                 color = 'red', line_width=3, alpha=0.5, 
                 legend='Training')
    hwplot.line(hw_forecasts.index.values, hw_forecasts[o], 
                 color = 'blue', line_width=3, line_dash='dashed',
                 alpha=0.5, legend='Forecast')
    hwplot.line(yact.index.values, yact[o], 
                 color = 'blue', line_width=3,
                 alpha=0.5, legend='Actual')
    hwplot.legend.location = 'top_left'
    hwplot.add_layout(Title(text="MAPE: "+ mape(yact[[o]], hw_forecasts[[o]]).astype(str), align="center"), "below")

    export_png(hwplot, filename='HW_Forecast_' + o + '.png')
    
for k in teams:
    fbplot = figure(title = 'Facebook Prophet Forecast -' + k, plot_width=800, plot_height=600, 
                     x_axis_type='datetime')
    fbplot.line(y1.index.values, y1[k], 
                 color = 'red', line_width=3, alpha=0.5, 
                 legend='Training')
    fbplot.line(fb_forecasts.index.values, fb_forecasts[k], 
                 color = 'blue', line_width=3, line_dash='dashed',
                 alpha=0.5, legend='Forecast')
    fbplot.line(yact.index.values, yact[k], 
                 color = 'blue', line_width=3,
                 alpha=0.5, legend='Actual')
    fbplot.legend.location = 'top_left'
    fbplot.add_layout(Title(text="MAPE: "+ mape(yact[[k]], fb_forecasts[[k]]).astype(str), align="center"), "below")

    export_png(fbplot, filename='FB_Forecast_' + k + '.png')
