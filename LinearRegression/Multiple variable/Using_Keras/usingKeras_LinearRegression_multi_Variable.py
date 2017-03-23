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

def Normalization(xtrain,m):
	print xtrain.shape
	print m
	for k in range(0,5):
		min = np.amin(xtrain[:,k])
		max = np.amax(xtrain[:,k])
		std = max-min
		mean = np.sum(xtrain[:,k])/(m)
		# print std
		# print mean
		xtrain[:,k] = (xtrain[:,k] - mean)/std

x = pd.ExcelFile("../data_carsmall.xlsx")
df = x.parse()
NaNIndex = df.index[df.y.isnull()]
testData = np.c_[df.x1[NaNIndex], df.x2[NaNIndex],df.x3[NaNIndex],df.x4[NaNIndex], df.x5[NaNIndex]]
# print testData
# print df.head()
df = df.dropna()
xtrain = np.c_[df.x1[1:], df.x2[1:], df.x3[1:], df.x4[1:], df.x5[1:]]
print xtrain.shape
m = len(xtrain)

# Normalization of all the feature using (feature - mean(features))std
Normalization(xtrain,m)
# for k in range(0,5):
# 	min = np.amin(xtrain[:,k])
# 	max = np.amax(xtrain[:,k])
# 	std = max-min
# 	mean = np.sum(xtrain[:,k])/(m)
# 	# print std
# 	# print mean
# 	xtrain[:,k] = (xtrain[:,k] - mean)/std
#print xtrain
xtrain = np.c_[np.ones(m), xtrain]
#plt.plot(xtrain)
print xtrain
# plt.plot(xtrain)
ytrain = df.y[1:]
#plt.plot(xtrain,ytrain)
#plt.plot(xtrain, ytrain, '*')
# plt.show()

##############################################################
#create Model
	
model = Sequential()
layer1 = Dense(6, input_shape = (6,))
model.add(layer1)
layer2 = Dense(1)
model.add(layer2)

model.compile(loss='mean_squared_error', optimizer='sgd',metrics=['accuracy'])

model.fit(xtrain, ytrain, nb_epoch=400, verbose=1)

####################################################################S
# xy = xtrain[0]
c = layer2.get_weights()
c = np.array(c)
print c[0]
# hypo = [[np.linspace(0,10,100),np.linspace(0,10,100),np.linspace(0,10,100),np.linspace(0,10,100),np.linspace(0,10,100)]]
# hypo = np.array(hypo)
# print c
# print hypo
# plt.plot(c.dot(hypo))
tSize = len(testData)
Normalization(testData, tSize)
# for t in range(0,5):
# 	min = np.amin(testData[:,t])
# 	max = np.amax(testData[:,t])
# 	std = max-min
# 	#print std
# 	mean = np.sum(testData[:,t])/(tSize)
# 	# print std
# 	# print mean
# 	testData[:,t] = (testData[:,t] - mean)/(std)
testData = np.c_[np.ones(tSize), testData]
s = [np.ones(100), np.linspace(-25,25,100), np.linspace(-25,25,100),np.linspace(-25,25,100),np.linspace(-25,25,100),
np.linspace(-25,25,100)]
s = np.array(s).transpose()
print s
print s.shape
plt.plot(s.dot(c[0]))
print testData
#z = model.predict(np.array([xy]))
z = model.predict(testData)
print z
plt.show()