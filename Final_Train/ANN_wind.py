#Full Wind Data model Training
import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

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

# Load train set
train_size2 = int(len(dataset2))
train2= dataset2[0:train_size2, :]

# reshape into X=t and Y=t+1
look_back = 1
trainX2, trainY2 = create_dataset(train2, look_back)

# reshape input to be [samples, time steps, features]
trainX2 = numpy.reshape(trainX2, (trainX2.shape[0], 1, trainX2.shape[1]))

# create and fit the LSTM network
model2 = Sequential()
model2.add(LSTM(4, input_shape=(1, look_back)))
model2.add(Dense(1))
model2.compile(loss='mean_squared_error', optimizer='adam')
model2.fit(trainX2, trainY2, epochs=100, batch_size=100, verbose=2)

# make predictions
trainPredict2 = model2.predict(trainX2)

# save the model
model2.save('wind.h5')
