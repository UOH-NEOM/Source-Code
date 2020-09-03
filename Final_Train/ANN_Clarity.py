#Full Clarity Data model Training
import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

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


train_size3 = int(len(dataset3))
train3= dataset3[0:train_size3, :]
# reshape into X=t and Y=t+1
look_back = 1
trainX3, trainY3 = create_dataset(train3, look_back)
# reshape input to be [samples, time steps, features]
trainX3 = numpy.reshape(trainX3, (trainX3.shape[0], 1, trainX3.shape[1]))
# create and fit the LSTM network
model3 = Sequential()
model3.add(LSTM(4, input_shape=(1, look_back)))
model3.add(Dense(1))
model3.compile(loss='mean_squared_error', optimizer='adam')
model3.fit(trainX3, trainY3, epochs=100, batch_size=100, verbose=2)

# make predictions
trainPredict3 = model3.predict(trainX3)

# save the model
model3.save('clarity.h5')
