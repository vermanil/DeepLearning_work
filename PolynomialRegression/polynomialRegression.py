'''
Created by vermanil
Polynomial Regression of multiple variable with Neural Network
data set are data_carsmall.xlsx

'''

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras import optimizers
import numpy as np
from keras import backend as K	
import matplotlib.pyplot as plt

def Normalization(xtrain,m):
	print xtrain.shape
	print m
	for k in range(0,20):
		min = np.amin(xtrain[:,k])
		max = np.amax(xtrain[:,k])
		std = max-min
		mean = np.sum(xtrain[:,k])/(m)
		# print std
		# print mean
		xtrain[:,k] = (xtrain[:,k] - mean)/std


import pandas as pd
x = pd.ExcelFile("data_carsmall.xlsx")
df = x.parse()
NaNIndex = df.index[df.y.isnull()]
testData = np.c_[df.x1[NaNIndex], df.x2[NaNIndex],df.x3[NaNIndex],df.x4[NaNIndex], df.x5[NaNIndex],df.x1[NaNIndex]*df.x2[NaNIndex],
df.x1[NaNIndex]*df.x3[NaNIndex],df.x1[NaNIndex]*df.x4[NaNIndex],df.x1[NaNIndex]*df.x5[NaNIndex],df.x2[NaNIndex]*df.x3[NaNIndex],df.x2[NaNIndex]*df.x4[NaNIndex],
df.x2[NaNIndex]*df.x5[NaNIndex],df.x3[NaNIndex]*df.x4[NaNIndex],df.x3[NaNIndex]*df.x5[NaNIndex],df.x4[NaNIndex]*df.x5[NaNIndex],
df.x1[NaNIndex]*df.x1[NaNIndex],df.x2[NaNIndex]*df.x2[NaNIndex],df.x3[NaNIndex]*df.x3[NaNIndex],df.x4[NaNIndex]*df.x4[NaNIndex],
df.x5[NaNIndex]*df.x5[NaNIndex]]
# print testData
print testData.shape
# print df.head()
df = df.dropna()
xtrain = np.c_[df.x1[1:], df.x2[1:], df.x3[1:], df.x4[1:], df.x5[1:],df.x1[1:]*df.x2[1:],
df.x1[1:]*df.x3[1:],df.x1[1:]*df.x4[1:],df.x1[1:]*df.x5[1:],df.x2[1:]*df.x3[1:],df.x2[1:]*df.x4[1:],
df.x2[1:]*df.x5[1:],df.x3[1:]*df.x4[1:],df.x3[1:]*df.x5[1:],df.x4[1:]*df.x5[1:],
df.x1[1:]*df.x1[1:],df.x2[1:]*df.x2[1:],df.x3[1:]*df.x3[1:],df.x4[1:]*df.x4[1:],
df.x5[1:]*df.x5[1:]]
print xtrain.shape
# print df.x1[1:]*df.x2[1:]
#plt.plot(xtrain,'*')
m = len(xtrain)
Normalization(xtrain,m)
xtrain = np.c_[np.ones(m), xtrain]
print xtrain.shape
l = len(testData)
Normalization(testData,l)
testData = np.c_[np.ones(l), testData]
# print testData
print testData.shape

ytrain = df.y[1:]

#############################################################
#Creating model

model = Sequential()
layer1 = Dense(21, input_shape = (21,))
model.add(layer1)

layer2 = Dense(1)
model.add(layer2)



model.compile(loss='mean_squared_error', optimizer='sgd',metrics=['accuracy'])

model.fit(xtrain, ytrain, nb_epoch=400, verbose=1)
c = layer2.get_weights()
c = np.array(c)
print c[0].shape
pl = np.c_[np.ones(93),df.x1[1:], df.x2[1:], df.x3[1:], df.x4[1:], df.x5[1:],df.x1[1:]*df.x2[1:],
df.x1[1:]*df.x3[1:],df.x1[1:]*df.x4[1:],df.x1[1:]*df.x5[1:],df.x2[1:]*df.x3[1:],df.x2[1:]*df.x4[1:],
df.x2[1:]*df.x5[1:],df.x3[1:]*df.x4[1:],df.x3[1:]*df.x5[1:],df.x4[1:]*df.x5[1:],
df.x1[1:]*df.x1[1:],df.x2[1:]*df.x2[1:],df.x3[1:]*df.x3[1:],df.x4[1:]*df.x4[1:],
df.x5[1:]*df.x5[1:]]
plt.plot(pl.dot(c[0]))

z = model.predict(testData)
print z
plt.show()
