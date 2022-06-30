import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

x = []
y = []
z = []

inputfile = "Y2_4KA.csv"
x_index = 0
y_index = 0
f = open(inputfile, 'r')
rdr = csv.reader(f)

#df = pd.read_csv(inputfile, header=None)

for line in rdr:
    for indentation in line:
        x.append(x_index)
        y.append(y_index)
        z.append(float(indentation))
        x_index = x_index + 1
    x_index = 0
    y_index = y_index + 1

x = np.array(x)
x.shape = (630000,1)
y = np.array(y)
y.shape = (630000,1)
z = np.array(z)
z.shape = (630000,1)

'''
for j in np.arange(0 ,31.4, 0.1):
    for i in np.arange(0, 20, 0.01):
        x.append(i)
        y.append(j)
        z.append( df )
        x_index =+ 1
    y_index = + 1
'''


'''
# Create the data.
from numpy import pi, sin, cos, mgrid
dphi, dtheta = pi/250.0, pi/250.0
[phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
m0 = 4; m1 = 3; m2 = 2; m3 = 3; m4 = 6; m5 = 2; m6 = 6; m7 = 4;
r = sin(m0*phi)**m1 + cos(m2*phi)**m3 + sin(m4*theta)**m5 + cos(m6*theta)**m7
x = r*sin(phi)*cos(theta)
y = r*cos(phi)
z = r*sin(phi)*sin(theta)
'''


# Figure
fig = plt.figure(figsize = (8, 8))

# 3DAxes
ax = fig.add_subplot(111, projection='3d')

# Axes
ax.set_title("", size = 20)

#
ax.set_xlabel("x", size = 14, color = "r")
ax.set_ylabel("y", size = 14, color = "r")
ax.set_zlabel("z", size = 14, color = "r")

#
#ax.set_xticks([-5.0, -2.5, 0.0, 2.5, 5.0])
#ax.set_yticks([-5.0, -2.5, 0.0, 2.5, 5.0])

# -5～5
#x = 10 * np.random.rand(100, 1) - 5
#y = 10 * np.random.rand(100, 1) - 5
#z = 10 * np.random.rand(100, 1) - 5

#
ax.scatter(x, y, z, s = 0.1, c = "blue")

plt.show()

'''
	X_index_list = ['X' + str(i + 1) for i in range(11)]
	X_index_list.append('ND')
	X_df = X_df[X_index_list]

	# return the data frame
	return X_df
'''