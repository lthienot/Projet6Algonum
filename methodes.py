import numpy as np
import math as m
import matplotlib.pyplot as plt

def step_euler(y, t, h, f):
    """ Calcule un pas avec la methode d'Euler """
    return y + h * f(y, t)


def step_middlepoint(y, t, h, f) :
    """ Calcule un pas avec la methode du point milieu """ 
    yinter = y + (h / float(2)) * f(y, t)
    pn = f(yinter, t + (h / float(2)))
    return y + h * pn

def step_heun(y, t, h, f) :
    """ Calcule un pas avec la methode de heun """
    pn1 = f(y, t)
    y1 = y + h * pn1
    pn2 = f(y1, t + h)
    ynplus1 = y + (float(1) / float(2)) * h * (pn1 + pn2)
    return ynplus1

def step_rk4(y, t, h, f):
    """ Calcule un pas avec la methode de R-K. 4 """
    pn1 = f(y,t)
    y1 = y + pn1 * h / float(2)
    pn2 = f(y1, t + h / float(2))
    y2 = y + pn2 * h / float(2)
    p3 = f(y2, t + h / float(2))
    y3 = y + p3 * h 
    p4 = f(y3, t + h)
    return y + (pn1 + 2 * pn2 + 2 * p3 + p4) * h / float(6)

def meth_n_step(y0, t0, N, h, f, meth):
    """ Etant donne un point (t0, y0), cette fonction\
    calcule un nombre N de pas (uniforfmement repartis)\
    de taille h en utilisant la methode meth pour le pb\
    de Cauchy represultente par y(t0) = y0, y'' = f(y)."""
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
    """ Calcule la norme infinie de (sol1 - sol2)\
    sol2 a deux fois plus de points que sol1 """
    res = 0
    for i in range(len(sol1)):
        if (res < np.linalg.norm(sol1[i]-sol2[2*i])):
            res = np.linalg.norm(sol1[i]-sol2[2*i])
    return res

def meth_epsilon(y0, t0, tf, eps, f, meth):
    """meth_epsilon retourne tableau solution au pb de Cauchy :\
    y(t0) = y0, y' = f(y)\
    Les valeurs sont uniformement reparties sur [t0,tf]\
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

def distinction(y):
    """ A partir de y,\
    renvoie le tableau des y[0],\
    et le tableau des y[1] """
    y_len = len(y)
    y0 = np.zeros(y_len)
    y1 = np.zeros(y_len)
    for i in range(y_len):
        y0[i] = y[i][0]
        y1[i] = y[i][1]
    return (y0, y1)    

def test_methods():
    """test_methods permet de verifier les resultats
    renvoyes par les methodes implementees plus haut
    sur des cas simples."""
    eps = 0.01
    tf = 8.

    #############################################
    ## Tests sur l'exponentielle
    #############################################
    
    exp_euler = meth_epsilon(np.array([1.]), 0., tf, eps, lambda y, t : y, "euler")
    exp_heun = meth_epsilon(np.array([1.]), 0., tf, eps, lambda y, t : y, "heun")
    exp_mdlpt = meth_epsilon(np.array([1.]), 0., tf, eps, lambda y, t : y, "middlepoint")
    exp_rk4 = meth_epsilon(np.array([1.]), 0., tf, eps, lambda y, t : y, "rk4")

    ##Calculons les tableaux d'abscisses
    x_euler=[0]
    for i in range(1,len(exp_euler)):
        x_euler += [ x_euler[i-1] + tf/float(len(exp_euler)) ]

    x_heun=[0]
    for i in range(1,len(exp_heun)):
        x_heun += [ x_heun[i-1] + tf/float(len(exp_heun)) ]

    x_mdlpt=[0]
    for i in range(1,len(exp_mdlpt)):
        x_mdlpt+= [ x_mdlpt[i-1] + tf/float(len(exp_mdlpt)) ]

    x_rk4=[0]
    for i in range(1,len(exp_rk4)):
        x_rk4 += [ x_rk4[i-1] + tf/float(len(exp_rk4)) ]

    # exp_math = [1]
    # len_x_exp = len(exp_euler)
    # for i in range(1, len_x_exp):
    #     exp_math += [ m.exp( x_euler[i] ) ]

    # Affichons
    plt.title("Comparaison de differentes approx de l'exponentielle")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x_euler,exp_euler,label='Euler', linewidth=1.0)
    plt.plot(x_heun ,exp_heun, label='Heun', linewidth=1.0)
    plt.plot(x_mdlpt ,exp_mdlpt, label='Middlepoint', linewidth=1.0)
    plt.plot(x_rk4 ,exp_rk4, label='RK4', linewidth=1.0)
#    plt.plot(x_euler ,exp_math, label='Bibliotheque math', linewidth=1.0)
#    plt.axis([4,tf,40,170])
    plt.legend()
    plt.show()

    print("Test exponentielle ok.")

    ###########################################
    ## Tests de l'enonce.
    ###########################################

    eps = 0.01

    ### Testons pour y(0)=1 et y'(t) = y(t) / 1 - t^2
    exp_rk41 =  meth_epsilon(np.array([1]), 0, 0.95, eps, lambda y, t : y[0]/float(1-t**2), "rk4")

    x_rk41=[0]
    for i in range(1,len(exp_rk41)):
        x_rk41 += [ x_rk41[i-1] + 0.99/float(len(exp_rk41)) ]


    # ### Testons pour y(0)=(1 0) et y'(t) = (-y2(t) y1(t))
    exp_rk42 =  meth_epsilon(np.array([1, 0]), 0., 0.95, eps, lambda y, t : np.array([-y[1], y[0]]), "rk4")

    (y1,y2) = distinction(exp_rk42)
    x_rk42=[0]
    for i in range(1,len(exp_rk42)):
        x_rk42 += [ x_rk42[i-1] + 1./float(len(exp_rk42)) ]

    ## Puis affichons : 
    plt.title("Affichage exemples de l'enonce (dim 1 et 2)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x_rk42,exp_rk41,label="Ex dim1 (cf enonce)", linewidth=1.0)
    plt.plot(x_rk41,y1, label="y1 de l'ex en dim2 (cd enonce)", linewidth=1.0)
    plt.plot(x_rk41,y2, label="y2 de l'ex en dim2 (cd enonce)", linewidth=1.0)
    plt.legend()
    plt.axis([0,0.95,0,6])
    plt.show()

    print("Test dim 2 ok.")
    


def main():
    test_methods()


if __name__ ==  '__main__':
    main()
