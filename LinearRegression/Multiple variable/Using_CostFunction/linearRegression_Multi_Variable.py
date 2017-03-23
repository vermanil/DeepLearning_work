'''
Created by vermanil
Linear Regression of multiple variable with cost function
data set are data_carsmall.xlsx

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = pd.ExcelFile("../data_carsmall.xlsx")
df = x.parse()
iter = 100
alpha = 0.01
# print df.head()
NaNIndex = df.index[df.y.isnull()]
testData = np.c_[df.x1[NaNIndex], df.x2[NaNIndex],df.x3[NaNIndex],df.x4[NaNIndex], df.x5[NaNIndex]]
df = df.dropna()
xtrain = np.c_[df.x1[1:], df.x2[1:], df.x3[1:], df.x4[1:], df.x5[1:]]
m = len(xtrain)
# print xtrain
# print xtrain[:,0]
for k in range(0,5):
	min = np.amin(xtrain[:,k])
	max = np.amax(xtrain[:,k])
	std = max-min
	mean = np.sum(xtrain[:,k])/(m)
	# print std
	# print mean
	xtrain[:,k] = (xtrain[:,k] - mean)/std
	# print xtrain[:,k]

# print m
# print xtrain
xtrain = np.c_[np.ones(m), xtrain]
# print xtrain
#plt.plot(xtrain)

test = xtrain[1,:]
ytrain = df.y[1:] 
# plt.figure()
# plt.hold(True)

theta = np.random.rand(1,6)[0]
for eochs in range(1000):
	cost = xtrain.dot(theta)-ytrain
	# print cost
	for i in range(0,6):
		theta[i] = theta[i] - alpha * np.sum(cost * xtrain[:, i])/(m)

tSize = len(testData)
for t in range(0,5):
 	min = np.amin(testData[:,t])
 	max = np.amax(testData[:,t])
 	std = max-min
	#print std
	mean = np.sum(testData[:,t])/(tSize)
# 	# print std
# 	# print mean
 	testData[:,t] = (testData[:,t] - mean)/(std)
testData = np.c_[np.ones(tSize), testData]
print testData
print "Optimize weights" 
print theta
s = [np.ones(100), np.linspace(-25,25,100), np.linspace(-25,25,100),np.linspace(-25,25,100),np.linspace(-25,25,100),
np.linspace(-25,25,100)]
s = np.array(s).transpose()

plt.plot(s.dot(theta))

out = testData.dot(theta)
print "predicted value"
print out
plt.show()

