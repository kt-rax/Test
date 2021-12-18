# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 20:51:35 2021

@author: KT
"""

'''
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation as amat

"function :f(x,y) = x^2 +y^2"
def GradFunction(x,y):
    return np.power(x,2) +np.power(y,2) 
def show(X,Y,func=GradFunction):
    fig = plt.figure()
    ax = Axes3D(fig)
    X,Y = np.meshgrid(X,Y,sparse=True)
    Z = func(X,Y)
    plt.title('grade')
    ax.plot_surface(X, Y, Z, rstride=1,cstride=1,cmap='rainbow')
    ax.set_xlabel('xlabel', color ='r')
    ax.set_ylabel('ylabel', color ='g')
    ax.set_zlabel('zlabel', color ='b')
    amat.FuncAnimation(fig, GradFunction, frames=200,interval=20,blit=True)
    plt.show()
    
if __name__ == '__main__':
    X = np.arange(-1.5,1.5,0.1)
    Y = np.arange(-1.5,1.5,0.1)
    Z = GradFunction(X, Y)
    show(X,Y,GradFunction)
'''

'''
import numpy as np
from matplotlib import pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation as amat

def GradFunction(x,y):
    return np.power(x,2)+np.power(y,2)

def show(X,Y,func=GradFunction):
    fig = plt.figure()
    ax = Axes3D(fig)
    X,Y= np.meshgrid(X, Y,sparse=True)
    Z = func(X, Y)
    plt.title('gradeAscent image')
    ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='rainbow')
    ax.set_xlabel('x label',color='r')
    ax.set_ylabel('y label',color='g')
    ax.set_zlabel('z label',color='b')
    plt.show()
    
def drawPaht(px,py,pz,X,Y,func=GradFunction):
    fig = plt.figure()
    ax = Axes3D(fig)
    X,Y = np.meshgrid(X, Y,sparse=True)
    Z = func(X,Y)
    plt.title('gradeAscent image')
    ax.set_xlabel('x label',color='r')
    ax.set_ylabel('y label',color='g')
    ax.set_zlabel('z label',color='b')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='raibow')
    plt.show()

def gradAscent(X,Y,Maxcycles=100,learnRate=0.1):
    new_x = [X]
    new_y = [Y]
    g_z = [GradFunction(X, Y)]
    current_x = X
    current_y = Y
    for cycle in range(Maxcycles):
        current_x -= learnRate*2*Y
        current_y -= learnRate*2*X
        X = current_x
        Y = current_y
        new_x.append(X)
        new_y.append(Y)
        g_z.append(GradFunction(X, Y))
    return new_x,new_y,g_z

if __name__ == '__main__':
    X = np.arange(-1,1,0.1)
    Y = np.arange(-1,1,0.1)
    x = 5
    y = 3
    print(x,y)
    x,y,z = gradAscent(x, y)
    print(x,y,z)
    drawPaht(x, y, z, X, Y,GradFunction)
'''
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
from sknn.mlp import Regressor
from sknn.mlp import Layer

hiddenLayer = Layer('Rectifier',units=6)
outputLayer = Layer('Linear',units=1)
nn = Regressor([hiddenLayer,outputLayer], learning_rule='sgd',learning_rate=.001,batch_size=5,loss_type='mse')

def cubic(X):
    return 2*X**3-3*X**2-X-1

def get_cubic_data(start,end,steo_size):
    X = np.arange(start,end,steo_size )
    X.shape = (len(X),1)
    y= np.array([cubic(X[i]) for i in range(len(X))])
    y.shape=(len(y),1)
    return X,y
    
X,Y = get_cubic_data(-2, 3, .01)
nn.fit(X,Y)
prediction = nn.predict(X)

plt.plot(prediction)
plt.plot(Y)
plt.show()
#plot_model(nn,to_file='./model1.png',show_shapes=True)

































