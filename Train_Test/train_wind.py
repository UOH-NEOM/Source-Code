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
	dataX2, dataY2 = [], []
	for i in range(len(dataset)-look_back-1):
		a2 = dataset[i:(i+look_back), 0]
		dataX2.append(a2)
		dataY2.append(dataset[i + look_back, 0])
	return numpy.array(dataX2), numpy.array(dataY2)

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe2 = read_csv('Wind.csv', usecols=[1], engine='python')
dataset2 = dataframe2.values
dataset2 = dataset2.astype('float32')

# normalize the dataset
scaler2 = MinMaxScaler(feature_range=(0, 1))
dataset2 = scaler2.fit_transform(dataset2)

# split into train and test sets
train_size2 = int(len(dataset2) * 0.67)
test_size2 = len(dataset2) - train_size2
train2, test2 = dataset2[0:train_size2, :], dataset2[train_size2:len(dataset2), :]

# reshape into X=t and Y=t+1
look_back = 1
trainX2, trainY2 = create_dataset(train2, look_back)
testX2, testY2 = create_dataset(test2, look_back)

# reshape input to be [samples, time steps, features]
trainX2 = numpy.reshape(trainX2, (trainX2.shape[0], 1, trainX2.shape[1]))
testX2 = numpy.reshape(testX2, (testX2.shape[0], 1, testX2.shape[1]))

# create and fit the LSTM network
model2 = Sequential()
model2.add(LSTM(4, input_shape=(1, look_back)))
model2.add(Dense(1))
model2.compile(loss='mean_squared_error', optimizer='adam')
model2.fit(trainX2, trainY2, epochs=100, batch_size=100, verbose=2)

# make predictions
trainPredict2 = model2.predict(trainX2)
testPredict2 = model2.predict(testX2)

# invert predictions
trainPredict2 = scaler2.inverse_transform(trainPredict2)
trainY2 = scaler2.inverse_transform([trainY2])
testPredict2 = scaler2.inverse_transform(testPredict2)
testY2 = scaler2.inverse_transform([testY2])

# calculate root mean squared error
trainScore2 = math.sqrt(mean_squared_error(trainY2[0], trainPredict2[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore2))
testScore2 = math.sqrt(mean_squared_error(testY2[0], testPredict2[:, 0]))
print('Test Score: %.2f RMSE' % (testScore2))

# shift train predictions for plotting
trainPredictPlot2 = numpy.empty_like(dataset2)
trainPredictPlot2[:, :] = numpy.nan
trainPredictPlot2[look_back:len(trainPredict2) + look_back, :] = trainPredict2

# shift test predictions for plotting
testPredictPlot2 = numpy.empty_like(dataset2)
testPredictPlot2[:, :] = numpy.nan
testPredictPlot2[len(trainPredict2) + (look_back * 2) + 1:len(dataset2) - 1, :] = testPredict2
