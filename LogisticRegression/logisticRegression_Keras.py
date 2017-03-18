'''
Created by vermanil
Logistic Regression Using Keras
Data set are ex2data1-logistic.xls and ex2data1-logistic.xls
'''
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras import optimizers
import pandas as pd
import numpy as np
x = pd.ExcelFile("ex2data1-logistic.xls")
df = x.parse()
#print df.head()
xtrain1 = np.c_[df.x1[5:95], df.x2[5:95]]
ytrain1 = df.y[5:95]
#print xtrain1.shape
#print test1
x1 = pd.ExcelFile("ex2data2-logistic.xls")
df1 = x1.parse()
#print df.head()
xtrain2 = np.c_[df1.x1[10:110], df1.x2[10:110]]
ytrain2 = df1.y[10:110]
#print xtrain2.shape
xtrain = np.concatenate((xtrain1,xtrain2))
#print xtrain.shape
test1 = np.concatenate((np.c_[df.x1[0:5], df.x2[0:5]], np.c_[df.x1[95:], df.x2[95:]]))
#print test1.shape
test2 = np.concatenate((np.c_[df1.x1[0:10], df1.x2[0:10]], np.c_[df1.x1[110:], df1.x2[110:]]))
# print test2.shape
testData = np.concatenate((test1,test2))
ytrain = np.concatenate((ytrain1, ytrain2))
print ytrain.shape
#print testData.shape
m1 = len(xtrain)
xtrain = np.c_[np.ones(m1), xtrain]
m2 = len(testData)
testData = np.c_[np.ones(m2), testData]
print xtrain.shape
print testData.shape

model = Sequential()
layer1 = Dense(3, input_shape = (3, ))
model.add(layer1)
#model.add(Activation('relu'))
layer2 = Dense(1)
model.add(layer2)
model.add(Activation('sigmoid'))

model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')

model.fit(xtrain, ytrain, verbose=1)

z = model.predict(testData)
print z
l = len(z)
for i in range(0,l):
	if z[i] >= 0.5:
		print "1"
	else:
		print "0"

