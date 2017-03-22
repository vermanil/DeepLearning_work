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
test = xtrain[1,:]
ytrain = df.y[1:] 
plt.plot(xtrain, ytrain)
theta = np.random.rand(1,6)[0]
for eochs in range(1000):
	cost = xtrain.dot(theta)-ytrain
	# print cost
	for i in range(0,6):
		theta[i] = theta[i] - alpha * np.sum(cost * xtrain[:, i])/(m)

print "Optimize weights" 
print theta
out = test.dot(theta)
print "predicted value"
print out
plt.show()

