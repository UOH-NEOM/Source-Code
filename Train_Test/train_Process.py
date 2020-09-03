#  Train/Test Compiled Process Model
from train_temp import  scaler, dataset, trainPredictPlot, testPredictPlot
from train_wind import scaler2, dataset2, trainPredictPlot2, testPredictPlot2
from train_clarity import scaler3, dataset3, trainPredictPlot3, testPredictPlot3
import matplotlib.pyplot as plt


# Gather all data frame in one plot
fig, axs = plt.subplots(3)
fig.suptitle('weather')
axs[0].plot(scaler.inverse_transform(dataset))
axs[0].plot(trainPredictPlot)
axs[0].plot(testPredictPlot)
axs[0].set_title("temprature")
axs[1].plot(scaler2.inverse_transform(dataset2))
axs[1].plot(trainPredictPlot2)
axs[1].plot(testPredictPlot2)
axs[1].set_title("wind")
axs[2].plot(scaler3.inverse_transform(dataset3))
axs[2].plot(trainPredictPlot3)
axs[2].plot(testPredictPlot3)
axs[2].set_title("Clarity")
plt.show()
