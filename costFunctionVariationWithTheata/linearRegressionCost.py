'''

Created by vermanil
Variation of Linear Regression cost function with Theta
Hypothesis is a sigmoid Function

'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
x = [79.0327360507,75.3443764369]
theta = np.c_[np.linspace(-25,25,100), np.linspace(-10,10,100)]
#print theta
# theta1 = np.linspace(-15,15,100)
# theta2 = np.linspace(-15,15,100)
z = theta[:,0] * x[0] + theta[:,1] * x[1]
#print np.exp(1)
# plt.plot(z)
htheta = (1)/(1+np.exp(-z))
print htheta.shape
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plt.plot(htheta)
xy = np.square(htheta)/(2)
print xy
ax.scatter(theta[:,0],theta[:,1], xy, '*')
xy1 = (-(np.log(htheta)))
print xy1
fig2 = plt.figure()
ax1 = fig2.add_subplot(111, projection='3d')
ax1.scatter(theta[:,0],theta[:,1], xy1, '++')
# plt.plot(np.square(htheta)/(2))
#plt.plot(-(np.log(1-htheta)))
# plt.plot(htheta)
# x = np.linspace(-50,50,100)
# htheta = (1)/(1+np.exp(-x))
# plt.plot(htheta)
plt.show()

 