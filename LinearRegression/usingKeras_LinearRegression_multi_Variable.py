'''
Created by vermanil
Linear Regression of multiple variable with Neural Network
data set are data_carsmall.xlsx

'''

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras import optimizers
import numpy as np
from keras import backend as K	
import matplotlib.pyplot as plt

import pandas as pd
x = pd.ExcelFile("data_carsmall.xlsx")
df = x.parse()
# print df.head()
df = df.dropna()
xtrain = np.c_[df.x1[1:], df.x2[1:], df.x3[1:], df.x4[1:], df.x5[1:]]
print xtrain.shape
m = len(xtrain)

# Normalization of all the feature using (feature - mean(features))std
for k in range(0,5):
	min = np.amin(xtrain[:,k])
	max = np.amax(xtrain[:,k])
	std = max-min
	mean = np.sum(xtrain[:,k])/(m)
	# print std
	# print mean
	xtrain[:,k] = (xtrain[:,k] - mean)/std
#print xtrain
xtrain = np.c_[np.ones(m), xtrain]
ytrain = df.y[1:]
# plt.plot(xtrain, ytrain, '*')
# plt.show()
	
model = Sequential()
layer1 = Dense(5, input_shape = (6,))
model.add(layer1)

layer2 = Dense(1)
model.add(layer2)


model.compile(loss='mean_squared_error', optimizer='sgd')
model.fit(xtrain, ytrain, verbose=1)


xy = xtrain[0]
c = layer2.get_weights()

z = model.predict(np.array([xy]))
print z