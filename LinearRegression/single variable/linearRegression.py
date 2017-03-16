'''
Created by vermanil
Linear Regression with cost function of single variable
dataset are in ex1data1.txt
'''
import numpy as np 
import matplotlib.pyplot as plt
import math 

#load the data
data = np.loadtxt('ex1data1.txt', delimiter=',')
itr = 1000
alpha = 0.01
x = data[:,0]
y = data[:,1]		
m = len(x)
data_new=np.c_[np.ones(m),x]
print data_new[:,0]
#theta = np.random.rand(1,2)[0]
#print theta[0]
#theta = np.zeros(2,1)[0]
theta = [0,0]
# a=data_new.dot(theta)
# d = a-y
# c = np.square(d)
# s = np.sum(c)
# print s
initialCost = np.sum(np.square(data_new.dot(theta)-y))/(2*m)
# print theta
# print yer
# f = float(s/(2*m))
# print a * s
# initialCost = f
print initialCost
for i in range(itr):
	cost = data_new.dot(theta)-y
	k = theta[0] - alpha * np.sum(cost * data_new[:,0])/(m)
	l = theta[1] - alpha * np.sum(cost * data_new[:,1])/(m)
	theta[0], theta[1] = k, l
print theta
predict = data_new.dot(theta)
plt.plot(x, predict)
# print z
plt.plot(x, y,'+')
plt.show()
