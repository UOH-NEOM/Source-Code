#Prediction model using Clarity samples from the dataset
import numpy
from pandas import read_csv
from keras.models import load_model

# convert an array of values into a dataset matrix
def create_dataset(dataset3, look_back=1):
	dataX3, dataY3 = [], []
	for i in range(len(dataset3) - look_back - 1):
		a3 = dataset3[i:(i + look_back), 0]
		dataX3.append(a3)
		dataY3.append(dataset3[i + look_back, 0])
	return numpy.array(dataX3), numpy.array(dataY3)
# fix random seed for reproducibility
numpy.random.seed(7)

# load the model
clarity_model = load_model('clarity.h5')

# load the dataset
dataframe3 = read_csv('Clarity Sample.csv', usecols=[1], engine='python')
dataset3 = dataframe3.values
dataset3 = dataset3.astype('float32')

# select the previous values
test3 = dataset3[69:, :]

# reshape into X=t and Y=t+1
look_back = 1
testX3, testY3 = create_dataset(test3, look_back)

# reshape input to be [samples, time steps, features]
testX3 = numpy.reshape(testX3, (testX3.shape[0], 1, testX3.shape[1]))

# make predictions
Clarity_Prediction = clarity_model.predict(testX3)
print(Clarity_Prediction)