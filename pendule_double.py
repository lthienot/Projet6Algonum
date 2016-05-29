import numpy as np
import math as m
import matplotlib.pyplot as plt
from methodes import *

l1 = 1
l2 = 1
m1 = 1
m2 = 1
t0 = 0.
tf = 10.
g = 9.81

def eq_pdouble():
    """Modelisation du pendule double"""

    return  lambda X, t: np.array([X[1],
                                   (-g*(2*m1+m2)*np.sin(X[0])-m2*g*np.sin(X[0]-2*X[2])-2*np.sin(X[0]-X[2])*m2*(l2*X[3]**2+l1*np.cos(X[0]-X[2])*X[1]**2))/(l1*(2*m1+m2-m2*np.cos(2*X[0]-2*X[2]))),
                                   X[3],
                                   (2*np.sin(X[0]-X[2])*((m1+m2)*l1*X[1]**2+g*(m1+m2)*np.cos(X[0])+l2*m2*np.cos(X[0]-X[2])*X[3]**2))/(l2*(2*m1+m2-m2*np.cos(2*X[0]-2*X[2])))])


def pos_pdouble(t0, y0, f):
    "Extremite du pendule double en fonction du temps"
    plt.clf()
    nmax = 1000
    res = meth_epsilon(y0, t0, tf, 10E-2, f,"rk4")
    y1 = []#theta1
    y2 = []#theta2
    x = []
    y = []

    for i in range(len(res)):
        y1 = y1 + [res[i][0]]
        y2 = y2 + [res[i][2]]

    for i in range(len(res)):
        x = x + [l1*np.sin(y1[i])-l2*np.sin(y2[i])]
        y = y + [-l1*np.cos(y1[i])-l2*np.cos(y2[i])]
    u = plt.plot(x, y, linewidth=1.0)

    plt.legend(u,"theta1 = Pi/5, theta2 = 2Pi/3")
    plt.show()

#pos_pdouble(t0, [np.pi/3, 0, 2*np.pi/3, 0], eq_pdouble())
