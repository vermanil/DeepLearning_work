import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-15,15,100)
y = np.r_[x * x + 1]
plt.plot(x,y,'x')

itr = int(input("enter the no of iteration!\n"))
alpha = float(input("enter the increment factor(ALPHA)!\n"))
point = [-50,-50]
plt.plot(point[0],point[1],'*')
for i in range(itr):
	point[0] = point[0] - alpha * (2*point[0])
	point[1] = point[1] - alpha * (2*point[1])
	plt.plot(point[0],point[1],'*')
plt.show()
#plt.contour(x,y)
