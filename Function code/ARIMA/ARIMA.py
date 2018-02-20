import pandas as pd
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
import sklearn
from sklearn.metrics import mean_squared_error
 
##prediction part

series = read_csv('top1restaurant.csv', header=0, parse_dates=[1], index_col=0, squeeze=True)
series = pd.DataFrame(series)
series.apply(lambda x: pd.to_numeric(x, errors='ignore'))
series.apply(lambda x: pd.to_datetime(x, errors='ignore'))
series=series.astype(float)
X = series.values
size = int(len(X) * 0.8)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

for t in range(200):
	model = ARIMA(history, order=(2,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	history.append(output[0])
pyplot.plot(predictions, color = 'red')
print (predictions[199])
pyplot.show()
'''
for t in range(len(test)):
	model = ARIMA(history, order=(2,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()

series = read_csv('top1restaurant.csv', header=0, parse_dates=[1], index_col=0, squeeze=True)
series = pd.DataFrame(series)
series.apply(lambda x: pd.to_numeric(x, errors='ignore'))
series.apply(lambda x: pd.to_datetime(x, errors='ignore'))
series=series.astype(float)
print(series.head())
series.plot()
#autocorrelation_plot(series)
pyplot.show()
# fit model
model = ARIMA(series, order=(0,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())'''