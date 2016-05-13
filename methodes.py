import numpy as np
import math
import matplotlib.pyplot as pyp

def step_euler(y, t, h, f):
    """ Calcule un pas avec la méthode d'Euler """
    return y + h * f(y, t)

def meth_n_step(y0, t0, N, h, f, meth):
    """ Etant donné un point (t0, y0), cette fonction calcule un nombre N de pas (uniforfmément répartis) de taille h en utilisant la méthode meth pour le pb de Cauchy represente par y(t0) = y0, y'' = f(y)."""
    y = y0
    t = t0
    result = [np.copy(y0)]
    for step in range(N):
        if (meth == "rk4"):
            y = step_rk4(y, t, h, f)
        elif (meth == "heun"):
            y = step_heun(y, t, h, f)
        elif (meth == "middlepoint"):
            y = step_middlepoint(y, t, h, f)
        elif (meth == "euler"):
            y = step_euler(y, t, h, f)
        else:
            raise("Error in the method name.")
           
        t = t + h
        result += [np.copy(y)]

    return result

# def meth_epsilon(y0, t0, tf, eps, f, meth):
#     """meth_epsilon retourne une table de valeurs,
#     uniformement repartis sur l'' intervalle [t0, tf]
#     resultution au probleme de cauchy
#     y(t0) = y0, y\' = f(y)"""
#     N = 16
#     done = False
#     prev_result = meth_n_step(y0, t0, 8, (tf-t0)/8, f, meth)
#     result = meth_n_step(y0, t0, 16, (tf-t0)/16, f, meth)
#     while (abs(np.norm(prev_result)-np.norm(result)) > eps):
#         prev_result = result
#         N = N * 2
#         h = (tf - t0) / N
#         result = meth_n_step(y0, t0, N, h, f, meth)
#     return result
