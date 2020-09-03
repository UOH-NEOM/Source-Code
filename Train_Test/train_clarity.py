# Train/Test Temprature model
import numpy
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX3, dataY3 = [], []
	for i in range(len(dataset)-look_back-1):
		a3 = dataset[i:(i+look_back), 0]
		dataX3.append(a3)
		dataY3.append(dataset[i + look_back, 0])
	return numpy.array(dataX3), numpy.array(dataY3)

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe3 = read_csv('Clarity.csv', usecols=[1], engine='python')
dataset3 = dataframe3.values
dataset3 = dataset3.astype('float32')

# normalize the dataset
scaler3 = MinMaxScaler(feature_range=(0, 1))
dataset3 = scaler3.fit_transform(dataset3)

# split into train and test sets
train_size3 = int(len(dataset3) * 0.67)
test_size3 = len(dataset3) - train_size3
train3, test3 = dataset3[0:train_size3, :], dataset3[train_size3:len(dataset3), :]

# reshape into X=t and Y=t+1
look_back = 1
trainX3, trainY3 = create_dataset(train3, look_back)
testX3, testY3 = create_dataset(test3, look_back)

# reshape input to be [samples, time steps, features]
trainX3 = numpy.reshape(trainX3, (trainX3.shape[0], 1, trainX3.shape[1]))
testX3 = numpy.reshape(testX3, (testX3.shape[0], 1, testX3.shape[1]))

# create and fit the LSTM network
model3 = Sequential()
model3.add(LSTM(4, input_shape=(1, look_back)))
model3.add(Dense(1))
model3.compile(loss='mean_squared_error', optimizer='adam')
model3.fit(trainX3, trainY3, epochs=100, batch_size=100, verbose=2)

# make predictions
trainPredict3 = model3.predict(trainX3)
testPredict3 = model3.predict(testX3)

# invert predictions
trainPredict3 = scaler3.inverse_transform(trainPredict3)
trainY3 = scaler3.inverse_transform([trainY3])
testPredict3 = scaler3.inverse_transform(testPredict3)
testY3 = scaler3.inverse_transform([testY3])

# calculate root mean squared error
trainScore3 = math.sqrt(mean_squared_error(trainY3[0], trainPredict3[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore3))
testScore3 = math.sqrt(mean_squared_error(testY3[0], testPredict3[:, 0]))
print('Test Score: %.2f RMSE' % (testScore3))

# shift train predictions for plotting
trainPredictPlot3 = numpy.empty_like(dataset3)
trainPredictPlot3[:, :] = numpy.nan
trainPredictPlot3[look_back:len(trainPredict3) + look_back, :] = trainPredict3

# shift test predictions for plotting
testPredictPlot3 = numpy.empty_like(dataset3)
testPredictPlot3[:, :] = numpy.nan
testPredictPlot3[len(trainPredict3) + (look_back * 2) + 1:len(dataset3) - 1, :] = testPredict3
