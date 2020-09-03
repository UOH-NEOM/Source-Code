# Prediction model using Temprature samples from the dataset
import numpy
from pandas import read_csv
from keras.models import load_model

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)

# load the model
temp_model = load_model('temp.h5')

# load the dataset
dataframe = read_csv('Temprature Sample.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

# select the previous values
test = dataset[69:,:]

# reshape into X=t and Y=t+1
look_back = 1
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# make predictions
Temp_Prediction = temp_model.predict(testX)
print(Temp_Prediction)
