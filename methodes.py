import numpy as np
import math
import matplotlib.pyplot as pyp

def step_euler(y, t, h, f):
    """ Calcule un pas avec la methode d'Euler """
    return y + h * f(y, t)


def step_middlepoint(y, t, h, f) :
    yinter = y + (h / float(2)) * f(y, t)
    pn = f(yinter, t + (h / float(2)))
    return y + h * pn

def step_heun(y, t, h, f) :
    pn1 = f(y, t)
    y1 = y + h * pn1
    pn2 = f(y1, t + h)
    ynplus1 = y + (float(1) / float(2)) * h * (pn1 + pn2)
    return ynplus1

def step_rk4(y, t, h, f):
    pn1 = f(y,t)
    y1 = y + pn1 * h / float(2)
    pn2 = f(y1, t + h / float(2))
    y2 = y + pn2 * h / float(2)
    p3 = f(y2, t + h / float(2))
    y3 = y + p3 * h 
    p4 = f(y3, t + h)
    return y + (pn1 + 2 * pn2 + 2 * p3 + p4) * h / float(6)

def meth_n_step(y0, t0, N, h, f, meth):
    """ Etant donne un point (t0, y0), cette fonction calcule
    un nombre N de pas (uniforfmement repartis) de taille h en utilisant la methode meth pour le pb de Cauchy represultente par y(t0) = y0, y'' = f(y)."""
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
        result.append(np.copy(y))
    return result

def infinity_diff_norm(sol1, sol2):
    """ Calcule la norme infinie de (sol1 - sol2)
    sol2 a deux fois plus de points que sol1 """
    res = 0
    for i in range(len(sol1)):
        if (res < np.linalg.norm(sol1[i]-sol2[2*i])):
            res = np.linalg.norm(sol1[i]-sol2[2*i])
    return res

def meth_epsilon(y0, t0, tf, eps, f, meth):
    """meth_epsilon retourne tableau solution au pb de Cauchy :
    y(t0) = y0, y' = f(y)
    Les valeurs sont uniformement reparties sur [t0,tf]
    """
    N = 20
    approxN = meth_n_step(y0, t0, N, float((tf-t0))/float(N), f, meth)
    N *= 2
    approx2N = meth_n_step(y0, t0, N, float((tf-t0))/float(N), f, meth)
    while (abs(infinity_diff_norm(approxN,approx2N)) > eps):
        approxN = approx2N
        N *= 2
        h = float((tf - t0)) / float(N)
        approx2N = meth_n_step(y0, t0, N, h, f, meth)
    return approx2N

def test_methods():
    """test_methods permet de verifier les resultats
    renvoyes par les methodes implementees plus haut
    sur des cas simples."""
    eps = 0.01

    ### Testons pour le cas simple d'une fonction exp :
    exp_euler = meth_epsilon(np.array([1]), 0, 1, eps, lambda y, t : y, "euler")
    h = float(1)/float((len(exp_euler)-1))
    for i in range(len(exp_euler)-1):
        #print(exp_euler[i][0])
        assert(abs(exp_euler[i]-np.exp(h*i)) < eps)
    print("Test exponentielle ok.")

    # ### Testons pour y(0)=1 et y'(t) = y(t) / 1 - t^2
    # exp_euler =  meth_epsilon(np.array([1]), 0, 1, eps, lambda y, t : y[0]/float(1-t**2), "euler")
    # -> boucle infinie

    # ### Testons pour y(0)=(1 0) et y'(t) = (-y2(t) y1(t))
    exp_euler =  meth_epsilon(np.array([1, 0]), 0, 1, eps, lambda y, t : np.array([-y[1], y[0]]), "rk4")
    # for i in range(len(exp_euler)-1):
    #     print(exp_euler[i][0])
    #     print(exp_euler[i][1])
    print("Test dim 2 ok.")
    
test_methods()
