#Prediction model using Wind samples from the dataset
import numpy
from pandas import read_csv
from keras.models import load_model

# convert an array of values into a dataset matrix
def create_dataset(dataset2, look_back=1):
	dataX2, dataY2 = [], []
	for i in range(len(dataset2) - look_back - 1):
		a2 = dataset2[i:(i + look_back), 0]
		dataX2.append(a2)
		dataY2.append(dataset2[i + look_back, 0])
	return numpy.array(dataX2), numpy.array(dataY2)

# fix random seed for reproducibility
numpy.random.seed(7)

# load the model
wind_model = load_model('wind.h5')

# load the dataset
dataframe2 = read_csv('Wind Speed Sample.csv', usecols=[1], engine='python')
dataset2 = dataframe2.values
dataset2 = dataset2.astype('float32')

# select the previous values
test2 = dataset2[69:, :]

# reshape into X=t and Y=t+1
look_back = 1
testX2, testY2 = create_dataset(test2, look_back)

# reshape input to be [samples, time steps, features]
testX2 = numpy.reshape(testX2, (testX2.shape[0], 1, testX2.shape[1]))

# make predictions
Wind_Prediction = wind_model.predict(testX2)
print(Wind_Prediction)