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
    res = meth_epsilon(y0, t0, tf, 10E-2, f,"rk4")
    res2 = meth_epsilon(y0+np.array([0.1, 0., 0., 0.]), t0, tf, 10E-2, f, "rk4")

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

    y1 = []#theta1
    y2 = []#theta2
    x = []
    y = []
    for i in range(len(res2)):
        y1 = y1 + [res2[i][0]]
        y2 = y2 + [res2[i][2]]
    for i in range(len(res2)):
        x = x + [l1*np.sin(y1[i])-l2*np.sin(y2[i])]
        y = y + [-l1*np.cos(y1[i])-l2*np.cos(y2[i])]
    v = plt.plot(x, y, linewidth=1.0)
    
    plt.legend((u,v),"theta1 = Pi/5, theta2 = 2Pi/3, theta1 = Pi/5+0.1, theta2 = 2Pi/3")
    plt.show()

def tmps_retour(x,res):
    """Retourne le temps du premier retournement a partir d'une courbe donnee"""
    i = 1
    while(np.pi-abs(res[i][0])>0. and np.pi-abs(res[i][2])>0. and i < len(res)-1):
        i = i+1
    return i
    
def afficher_tmps(n):
    """Graphe du temps mis par le pendule double pour se retourner"""
    nmax = 100
    tmax = 10.
    x = np.arange(0.,tmax+tmax/nmax,tmax/nmax)
    V = np.zeros([n,n])
    a = (2*np.pi)/V.shape[0]
    b = -np.pi
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            print ((i*V.shape[1]+j)/(float(V.shape[0]*V.shape[1])))*100.,"%"
            y0 = np.array([i*a+b, 0, j*a+b, 0.])
            res = meth_n_step(y0, t0, nmax, 10./nmax, eq_pdouble(), "rk4")
            tps = tmps_retour(x,res)
            if(tps == len(x)-1):
                V[i][j] = x[0]
            else:
                V[i][j] = x[tmps_retour(x,res)]
    plt.clf()
    plt.imshow(V)
    plt.gcf()
    plt.clim()
    plt.colorbar()
    plt.show()


####### TESTS #####

#pos_pdouble(t0, [np.pi/3, 0, 2*np.pi/3, 0], eq_pdouble())
#afficher_tmps(100)
