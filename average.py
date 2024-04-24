from pandas import read_csv
from pandas import to_datetime
from datetime import datetime
from matplotlib import pyplot
import matplotlib
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
from tqdm import tqdm_notebook
import warnings
warnings.filterwarnings('ignore')
matplotlib.use('TkAgg')

series = read_csv('fivecities.csv', header=0, parse_dates=[1], index_col=1)

beijing_data = series.loc[series['City'] == 'Beijing','PM_average'].resample('M').mean()
shanghai_data = series.loc[series['City'] == 'Shanghai','PM_average'].resample('M').mean()
guangzhou_data = series.loc[series['City'] == 'Guangzhou','PM_average'].resample('M').mean()
chengdu_data = series.loc[series['City'] == 'Chengdu','PM_average'].resample('M').mean()
shenyang_data = series.loc[series['City'] == 'Shenyang','PM_average'].resample('M').mean()
PMdata = (beijing_data + shanghai_data + guangzhou_data + chengdu_data + shenyang_data)/5


# PMdata.dropna().plot()
PMdata=PMdata.fillna(PMdata.mean())
# plot_acf(PMdata)
# plot_pacf(PMdata)
# pyplot.show()


from statsmodels.tsa.stattools import adfuller
result = adfuller(PMdata)
print( ' p-value: ', result[1])
result = adfuller(PMdata.diff().dropna())
print( ' p-value: ', result[1])
result = adfuller(PMdata.diff().diff().dropna())
print( 'p-value: ', result[1])

X = PMdata.values
size = int(len(X) *5/6)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
# walk-forward validation
for t in range(len(test)):

    model = SARIMAX(history, order=(0, 0, 0), seasonal_order = (2,2,0,12))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes

pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()

# from pmdarima.arima import auto_arima
# model1=auto_arima(PMdata,start_p=0,start_q=0,max_p=3,max_q=3,seasonal=False,d=2,trace = True,error_action ='ignore',suppress_warnings = True,stepwise=True)
# model2=auto_arima(PMdata,start_p=0,start_q=0,max_p=3,max_q=3,m=12,start_P=0,seasonal=True,d=0,D=0,trace = True,error_action ='ignore',suppress_warnings = True,stepwise=True)

model = SARIMAX(PMdata, order=(0, 0, 0), seasonal_order = (2,2,0,12)) #220是423 428,221是424 431
# model = SARIMAX(PMdata, order=(3, 0, 1), seasonal_order = (0,1,1,12))#496,509
# model = SARIMAX(PMdata, order=(1, 0, 0), seasonal_order = (2,0,0,12))#太高
# model = SARIMAX(PMdata, order=(0, 1, 1), seasonal_order = (1,2,0,12))#438,443
# model = SARIMAX(PMdata, order=(0, 1, 3), seasonal_order = (0,1,1,12))#494
# model = SARIMAX(PMdata, order=(1, 1, 0), seasonal_order = (2,0,0,12))#不行
# model = SARIMAX(PMdata, order=(3, 2, 0), seasonal_order = (2,2,1,12))#拔高,但看起来最靠谱439,452
# model = SARIMAX(PMdata, order=(3, 2, 0), seasonal_order = (2,1,0,12))#pass
# model = SARIMAX(PMdata, order=(3, 2, 0), seasonal_order = (1,0,0,12))#pass
model_arima = ARIMA(PMdata, order=(1,0,0))  #591,598
# model_arima = ARIMA(PMdata, order=(0,1,0))#不如上一个
# model_arima = ARIMA(PMdata, order=(2,2,0))#不如第一个
model_fit = model.fit()
model_arimafit = model_arima.fit()

# 打印模型摘要
print(model_fit.summary())
print(model_arimafit.summary())

# 预测未来一年的数据
forecast = model_fit.forecast(steps=12)
forecast_arima = model_arimafit.forecast(steps=12)
# 可视化预测结果

pyplot.figure(figsize=(12, 6))
pyplot.plot(PMdata, label='Original Data')
pyplot.plot(pd.date_range(start='2016-01', periods=12, freq='ME'), forecast, label='Forecast')
pyplot.title('SARIMA Forecast for Next Year')
pyplot.xlabel('Date')
pyplot.ylabel('Value')

pyplot.figure(figsize=(12, 6))
pyplot.plot(PMdata, label='Original Data')
pyplot.plot(pd.date_range(start='2016-01', periods=12, freq='ME'), forecast_arima, label='Forecast')
pyplot.title('ARIMA Forecast for Next Year')
pyplot.xlabel('Date')
pyplot.ylabel('Value')

model_fit.plot_diagnostics(figsize=(15,12))
model_arimafit.plot_diagnostics(figsize=(15,12))


pyplot.legend()
pyplot.show()




