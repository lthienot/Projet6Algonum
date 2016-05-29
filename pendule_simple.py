import numpy as np
import math as m
import matplotlib.pyplot as plt
from methodes import *

g = 9.81
t0 = 0.
tf = 10.
eps = 10E-1
l = 1

def eq_psimple():
    """ Modelise l'equation du mouvement du pendule simple"""
    return lambda X, t: np.array([X[1], (-g/l) * np.sin(X[0])])

def freq_psimple(theta):
    """Calcule la frequence du pendule simple pour un angle theta fourni en parametre"""
    y0 = np.array([theta, 0.])
    res = meth_epsilon(y0, t0, tf, eps, eq_psimple(), "rk4")
    
    i = 0
    deb_variation = abs(res[1][0] - res[0][0]) / (res[1][0] - res[0][0])
    variation = deb_variation
    while( (abs(res[i][1]) > eps or variation == deb_variation) and i < len(res)-1 ):
        i += 1
        variation = abs(res[i+1][0] - res[i][0]) / (res[i+1][0] - res[i][0])

    return 1/(t0 + i*eps)

def tracer_freq(min_theta, max_theta):
    """Trace la frequence du pendule simple de longueur l 
    en fonction de l'angle theta, entre min_theta et max_theta"""
    theta = np.arange(min_theta, max_theta, 10E-2)

    y = []
    for i in range(len(theta)):
        y = y + [freq_psimple(theta[i])]

    u = plt.plot(theta, y, linewidth=1.0)
    plt.xlabel("Theta")
    plt.ylim(0,0.6)
    plt.ylabel("Frequence")
    plt.show()

def pos_simple(theta):
    """On resout l'equation differentielle"""
    y0 = np.array([theta,0.])
    plt.clf()
    nmax = 300
    y1 = []
    y2 = []
    res = meth_epsilon(y0, t0, tf, 10E-3, eq_psimple(), "rk4")
    x = np.arange(t0, tf, (tf - t0)/len(res))
    for i in range(len(res)):
        y1 = y1 + [res[i][0]]
        y2 = y2 + [res[i][1]]
    u = plt.plot(x, y1, linewidth=1.0)
    v = plt.plot(x, y2, linewidth=1.0)
    plt.legend((u,v),("Theta","Omega"))
    plt.show()


########### TESTS ###########

tracer_freq(np.pi/20, np.pi/4)
pos_simple(np.pi/10)
