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

city_dict = {
    1: 'Beijing',
    2: 'Shanghai',
    3: 'Guangzhou',
    4: 'Chengdu',
    5: 'Shenyang',
    6: 'Average'
}

# 获取用户输入的城市名称
print("Please select the city number to search：")
for key, value in city_dict.items():
    print(f"{key}. {value}")

city_number = int(input("Please enter the selected city number："))

if city_number in city_dict:
    city_name = city_dict[city_number]
    if city_name == 'Average':
        # 执行 Average 数据的操作
        print("你选择了 Average 数据。")
        beijing_data = series.loc[series['City'] == 'Beijing','PM_average'].resample('M').mean()
        shanghai_data = series.loc[series['City'] == 'Shanghai','PM_average'].resample('M').mean()
        guangzhou_data = series.loc[series['City'] == 'Guangzhou','PM_average'].resample('M').mean()
        chengdu_data = series.loc[series['City'] == 'Chengdu','PM_average'].resample('M').mean()
        shenyang_data = series.loc[series['City'] == 'Shenyang','PM_average'].resample('M').mean()
        PMdata = (beijing_data + shanghai_data + guangzhou_data + chengdu_data + shenyang_data)/5

    else:
        PMdata = series.loc[series['City'] == city_name, 'PM_average'].resample('M').mean()
        print(PMdata.head())

    PMdata.dropna().plot()
    PMdata=PMdata.fillna(PMdata.mean())
    # PMdata= PMdata.dropna()
    plot_acf(PMdata)
    plot_pacf(PMdata)
    pyplot.show()
else:
    print("The number you entered was invalid. Please try again.")



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

model = SARIMAX(PMdata, order=(0, 0, 0), seasonal_order = (2,2,0,12)) #AIC: 492   497
# modmafit = model_arima.fit()el = SARIMAX(beijing_data, order=(2, 1, 0), seasonal_order = (2,2,1,12))#500 511
# model_arima = ARIMA(beijing_data, order=(2,0,2)) #AIC 677.026 BIC 690.686
model_arima = ARIMA(PMdata, order=(0,1,2))  #682    684.9
#上海的是(2,0,1)
#广州(0,0,1)
#成都(2,2,0)
#沈阳100
#均值100
# model_arima = ARIMA(beijing_data, order=(0,1,2)) # 678.427 685.215
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




