# Full training Compiled Process Model
from ANN_temp import model
from ANN_wind import model2
from ANN_Clarity import model3

#Random Values & Prediction to activate model varibles
X =[ [33], [32], [31]]
X2 =[ [6], [6.5], [6.3]]
X3 = [[1], [0.8], [0.9]]

Y = model.predict(X)
Y2 = model2.predict(X2)
Y3 = model3.predict(X3)

print(Y, Y2, Y3)
