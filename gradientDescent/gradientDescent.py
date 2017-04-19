'''
Created by vermanil
Greadent Descent of function f(x) = x^2 + 1

'''
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-15,15,100)
y = np.r_[x * x + 1]
plt.plot(x,y,'x')

alpha = float(input("enter the increment factor(ALPHA)!\n"))
point = [30,50]
plt.plot(point[0],point[1],'*')
#apply gradient descent
x = 0;
y= 0;
while(1):
	point[0] = point[0] - alpha * (2*point[0])
	point[1] = point[1] - alpha * (2*point[1])
	plt.plot(point[0],point[1],'*')
	print(abs(round(point[0] - x,5)))
	if abs(round(point[0] - x,4)) == 0.0001 or abs(round(point[1] - y,4)) == 0.0001:
		break;

	print point[0],point[1]
print "Final Optimum point"
print "%.2f" % point[0], "%.2f" % point[1]
plt.show()
#plt.contour(x,y)
