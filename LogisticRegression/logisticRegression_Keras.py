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
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


x = pd.ExcelFile("ex2data1-logistic.xls")
df = x.parse()
#print df.y[0]
#print df.head()
xtrain1 = np.c_[df.x1[5:95], df.x2[5:95]]
ytrain1 = df.y[5:95]
#print xtrain1.shape
#print test1
x1 = pd.ExcelFile("ex2data2-logistic.xls")
df1 = x1.parse()
# print df1
df1 = df1.sample(frac=1).reset_index(drop=True)
# print df1
#print df.head()
xtrain2 = np.c_[df1.x1[10:110], df1.x2[10:110]]
ytrain2 = df1.y[10:110]
#print xtrain2.shape
# xtrain = np.concatenate((xtrain1,xtrain2))
#print xtrain.shape
test1 = np.concatenate((np.c_[df.x1[0:5], df.x2[0:5]], np.c_[df.x1[95:], df.x2[95:]]))
actOutput1 = np.concatenate((np.c_[df.y[0:5]], np.c_[df.y[95:]]))
#print test1.shape
test2 = np.concatenate((np.c_[df1.x1[0:10], df1.x2[0:10]], np.c_[df1.x1[110:], df1.x2[110:]]))
actOutput2 = np.concatenate((np.c_[df1.y[0:10]], np.c_[df1.y[110:]]))



m1 = len(xtrain1)
xtrain1 = np.c_[np.ones(m1), xtrain1]
m2 = len(test1)
testData = np.c_[np.ones(m2), test1]


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Define Model

model = Sequential()
layer1 = Dense(3, input_shape = (3, ), init='normal')
model.add(layer1)


layer2 = Dense(1)
model.add(layer2)
model.add(Activation('sigmoid'))

adam = optimizers.Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.

model.fit(xtrain1, ytrain1, nb_epoch=1000, verbose=1)

c = layer2.get_weights()
c = np.array(c).transpose()
print c

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Train First Data

z = model.predict(testData)

print z
l = len(z)
outPut = []
for i in range(0,l):
	if z[i] >= 0.5:
		outPut.append(1)
	else:
		outPut.append(0)
outPut = np.array(outPut)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Train second Data
model1 = Sequential()
layer1 = Dense(3, input_shape = (3, ), init='normal')
model1.add(layer1)


layer2 = Dense(1)
model1.add(layer2)
model1.add(Activation('sigmoid'))

adam = optimizers.Adam(lr=0.001)
model1.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])


m1 = len(xtrain2)
xtrain2 = np.c_[np.ones(m1), xtrain2]
m2 = len(test2)
testData = np.c_[np.ones(m2), test2]

model1.fit(xtrain2, ytrain2, nb_epoch=1000, verbose=1)
c = layer2.get_weights()
c = np.array(c).transpose()
print c
z = model1.predict(testData)

print z
l = len(z)
outPut2 = []
for i in range(0,l):
	if z[i] >= 0.5:
		outPut2.append(1)
	else:
		outPut2.append(0)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
print "\n\n"
# print "Here Accuracy we got"
countFirst = 0
k = (len(actOutput1))
for i in range(0,k):
	if actOutput1[i,0] == outPut[i]:
		countFirst = countFirst + 1
# print countFirst
print "Here Accuracy we got ", float(countFirst)/float(k)
print "Pridicted Output for first dataset"
print outPut
print "Actual"
print actOutput1.transpose()


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



outPut2 = np.array(outPut2)
print "\n\n"
k = len(actOutput2)
# print actOutput2[1,0]
countSecond = 0
for i in range(0,k):
	if actOutput2[i,0] == outPut2[i]:
		countSecond = countSecond + 1
# print countSecond,k
# print float(10)/float(18)
print "Here Accuracy we got ", float(countSecond)/float(k)
print "Pridicted Output for second dataset"
print outPut2
print "Actual"
print actOutput2.transpose()
# print xtrain.shape
# print testData.shape
plt.show()


