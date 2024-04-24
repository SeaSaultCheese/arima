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

# 定义解析函数
# def parser(x):
#     return datetime.strptime(x, '%Y/%m/%d %H:%M').strftime('%Y-%m-%d')  # 将小时部分去掉
series = read_csv('beijing.csv', header=0, parse_dates=[1], index_col=1)
# date_parser=parser
print(series.head())
beijing_data = series['PM_average'].resample('M').mean()
beijing_data = beijing_data.fillna(beijing_data.mean())
# beijing_data.plot()

#
# # plot_acf(beijing_data.diff().dropna(),lags=35)
# # plot_pacf(beijing_data.diff().dropna(),lags=35)
plot_acf(beijing_data.dropna(),lags=35)
plot_pacf(beijing_data.dropna(),lags=35)
pyplot.show()

from statsmodels.tsa.stattools import adfuller
result = adfuller(beijing_data.dropna())
print( ' p-value: ', result[1])
result = adfuller(beijing_data.diff().dropna())
print( ' p-value: ', result[1])
result = adfuller(beijing_data.diff().diff().dropna())
print( 'p-value: ', result[1])


# split into train and test sets
X = beijing_data.values
size = int(len(X) *7/10)
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


from pmdarima.arima import auto_arima
# model1=auto_arima(beijing_data,start_p=0,start_q=0,max_p=3,max_q=3,seasonal=False,d=1,trace = True,error_action ='ignore',suppress_warnings = True,stepwise=True)
# model2=auto_arima(beijing_data,start_p=0,start_q=0,max_p=3,max_q=3,m=12,start_P=0,seasonal=True,d=1,D=2,trace = True,error_action ='ignore',suppress_warnings = True,stepwise=True)


#
# def optimize_SARIMA(parameters_list, d, D, s, exog):
#     """
#         Return dataframe with parameters, corresponding AIC and SSE
#
#         parameters_list - list with (p, q, P, Q) tuples
#         d - integration order
#         D - seasonal integration order
#         s - length of season
#         exog - the exogenous variable
#     """
#
#     results = []
#
#     for param in parameters_list:
#         try:
#             model = SARIMAX(exog, order=(param[0], d, param[1]), seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
#         except:
#             continue
#
#         aic = model.aic
#         results.append([param, aic])
#
#     result_df = pd.DataFrame(results)
#     result_df.columns = ['(p,q)x(P,Q)', 'AIC']
#     #Sort in ascending order, lower AIC is better
#     result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
#
#     return result_df
#
# p = range(0, 4, 1)
# d = 0
# q = range(0, 4, 1)
# P = range(0, 2, 1)
# D = 2
# Q = range(0, 2, 1)
# s = 12
# parameters = product(p, q, P, Q)
# parameters_list = list(parameters)
# print(len(parameters_list))
#
# result_df = optimize_SARIMA(parameters_list, 1, 0, 12, beijing_data)
# print(result_df)

# model = SARIMAX(beijing_data, order=(0, 1, 1), seasonal_order = (0,0,1,12))  #675,  682
# model = SARIMAX(beijing_data, order=(1,1,0),seasonal_order=(2,1,0,12)) #AIC 579.461 BIC 587.772
# model = SARIMAX(beijing_data, order=(0, 1, 1), seasonal_order = (0,1,1,12))#579.704 BIC 585.937不如
# model = SARIMAX(beijing_data, order=(1, 1, 1), seasonal_order = (1,0,1,12)) #AIC:673.669 BIC:684.167
# model = SARIMAX(beijing_data, order=(0, 0, 0), seasonal_order = (2,1,0,12)) #575 582比ARIMA歪
# model = SARIMAX(beijing_data, order=(2, 1, 0), seasonal_order = (2,2,1,12))#500 511拔高那个
# model = SARIMAX(beijing_data, order=(2, 2, 0), seasonal_order = (1,2,0,12))#500 511
# model_arima = ARIMA(beijing_data, order=(2,0,2)) #AIC 677.026 BIC 690.686
            # model_arima = ARIMA(beijing_data, order=(1,0,0))  #682    684.9

model = SARIMAX(beijing_data, order=(0, 0, 0), seasonal_order = (2,2,0,12)) #AIC: 492   497
model_arima = ARIMA(beijing_data, order=(0,1,2)) # 678.427 685.215

# model = SARIMAX(beijing_data, order=(0, 0, 1), seasonal_order = (1,0,1,12)) #AIC: 492   497
# model_arima = ARIMA(beijing_data, order=(0,0,1)) # 678.427 685.215



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
pyplot.plot(beijing_data, label='Original Data')
pyplot.plot(pd.date_range(start='2016-01', periods=12, freq='ME'), forecast, label='Forecast')
pyplot.title('SARIMA Forecast for Next Year')
pyplot.xlabel('Date')
pyplot.ylabel('Value')

pyplot.figure(figsize=(12, 6))
pyplot.plot(beijing_data, label='Original Data')
pyplot.plot(pd.date_range(start='2016-01', periods=12, freq='ME'), forecast_arima, label='Forecast')
pyplot.title('ARIMA Forecast for Next Year')
pyplot.xlabel('Date')
pyplot.ylabel('Value')

model_fit.plot_diagnostics(figsize=(15,12))
model_arimafit.plot_diagnostics(figsize=(15,12))


pyplot.legend()
pyplot.show()