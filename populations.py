from methodes import *
import numpy as np
import matplotlib.pyplot as plt
import math

# APPLICATION MODELISATION DE POPULATIONS

###############################################
## Outils : tabulation de x
##############################################

def x_values(x0, xf, N):
    """x_values renvoie le tableau de N valeurs de x \
    entre x0 et xf (uniformement reparties) """
    h = ( xf - x0 ) / ( N-1 )
    res = np.zeros(N)
    for i in range(N):
        res[i] = x0 + h * i
    return res

##############################################
## Modele Malthusien
#############################################

def modele_malthusien():
    """ modele_malthusien affiche differents resultats \
    obtenus en utilisant le modele de Malthus, et ce en \
    fonction du parametre gamma. """

    t0 = 0.      #intervalle d'affichage : [t0,tf]
    tf = 2.
    eps = 0.1    #precision
    nb_individus_init = np.array([1000.])#population initiale

    ## gamma = birth - death
    gamma = 2    
    augmentation_population = meth_epsilon(nb_individus_init, t0, tf, eps,\
                                           lambda y, t: gamma*y, "rk4");
    
    gamma = -2
    declin_population = meth_epsilon(nb_individus_init, t0, tf, eps,\
                                     lambda y, t: gamma*y, "rk4");

    gamma = 0
    population_constante = meth_epsilon(nb_individus_init, t0, tf, eps, \
                                        lambda y, t: gamma*y, "rk4");

    plt.plot(x_values(t0, tf, len(augmentation_population)), \
             augmentation_population, "green",\
             label="natalite > mortalite : AUGMENTATION")

    plt.plot(x_values(t0, tf, len(declin_population)), declin_population,\
             "purple", label="natalite < mortalite : DECLIN")

    plt.plot(x_values(t0, tf, len(population_constante)), population_constante,\
             "grey", label="natalite == mortalite : STAGNATION")

    plt.xlabel("temps t")
    plt.ylabel("nombre d'individus")
    plt.axis([0,2,0,5000])
    plt.legend()
    plt.title("Resultats du modele malthusien d'evolution d'une population")
    plt.show()

##########################################
## Modele de Verhulst
##########################################
    
def modele_de_verhulst():
    """ modele_de_verhulst affiche differents resultats \
    obtenus en utilisant le modele de Verhulst, et ce en \
    fonction des parametres gamma et k """
    
    t0 = 0.
    tf = 7.
    eps = 0.01
    nb_individus_init = np.array([1000.])

    gamma = 1 ##vitesse croissance

    k = 1500 ##max population
    pop_croissante = meth_epsilon(nb_individus_init, t0, tf, eps,\
                                  lambda y, t: gamma * y * (1 - y/k),\
                                  "rk4");

    k = 500
    pop_decroissante  = meth_epsilon(nb_individus_init, t0, tf, eps,\
                                     lambda y, t: gamma * y * (1 - y/k),\
                                     "rk4");

    k = 1000
    pop_cste = meth_epsilon(nb_individus_init, t0, tf, eps,\
                            lambda y, t: gamma * y * (1 - y/k),\
                            "rk4");

    #Affichage 
    plt.plot(x_values(t0, tf, len(pop_croissante)),pop_croissante,"green",
             label="nb individus init < nb limite (1500)")

    plt.plot(x_values(t0, tf, len(pop_decroissante)), pop_decroissante,"grey",
             label="nb individus init > nb limite (500)")

    plt.plot(x_values(t0, tf, len(pop_cste)), pop_cste, "purple", 
             label="nb individus init == nb limite")

    plt.xlabel("temps")
    plt.ylabel("nb d'individus")
    plt.axis([int(t0),int(tf),400,1800])
    plt.title("Resultats du modele de Verhulst d'evolution d'une population")
    plt.legend()
    plt.show()

#######################################################
## Modele de Lotka-Volterra
######################################################
    
def derivee_Nt_Pt(a, b, c, d):
    """ Systeme du modele de Lotka-Volterra\
    avec y[0] = N(t), y[1] = P(t), et avec \
    pour parametres : a,b,c,d > 0."""
    return lambda y, t : np.array([ y[0] * ( a - b * y[1]),\
                                    y[1] * ( c * y[0] - d)])


def approx_periode(y, t0, tf):
    "approx_periode calcule la periode de la solution y,\
    tabulee entre t=t0 et t=tf"""
    
    y_len = len(y)
    x = x_values(t0, tf, y_len)

    nb_periodes = 0
    max_periode_1 = 0
    max_derniere_periode = 0
    i = 1
    while (i < y_len): #Parcours des differentes periodes ... 
        while ((i < y_len) and (y[i-1] > y[i])):
            i += 1
        while((i < y_len) and (y[i-1] < y[i])):
            i += 1
        if (i < y_len):
            if (nb_periodes == 0):
                max_periode_1 = x[i-1]
            max_derniere_periode = x[i-1]
            nb_periodes += 1

    if (nb_periodes < 2):
        print("Calcul impossible : trop peu de periodes a disposition.")
        return 0
    else:
        return (max_derniere_periode-max_periode_1)/(nb_periodes-1)


def diagramme_solutions(nb_individus_init, var, N, a, b, c, d, t0, tf, eps):
    """ diagramme_solutions trace les solutions P(t)=f(N(t)),\
    avec pour origine nb_individus_init, puis en prenant \
    des entiers autour de nb_individus_init\
    -> donc variations du nombre de proies et de predateurs\
    Parametres :\
    - nb_individus_init (y0 dans enonce)\
    - var (entier)\
    - N (entier) -> pour nb de variations de (2N+1)*(2N+1) \
    -a,b,c,d (cf parametres lotka-volterra)\
    - t0, tf pr l'intervalle de temps\
    - eps pour la precision des solutiosn"""

    h = var/(2*N)
    
    for i in range (-N, N+1):
        for j in range (-N, N+1):
            nv_nb_individus = nb_individus_init + np.array([h*i,h*j])
            res_l_v = meth_epsilon(nv_nb_individus, t0, tf, eps,\
                                   derivee_Nt_Pt(a, b, c, d), "rk4");
            (Nt, Pt) = distinction_Nt_Pt(res_l_v)
            plt.plot(Nt, Pt)

    plt.xlabel("N(t), nb de proies")
    plt.ylabel("P(t), nb de predateurs")
    plt.title("ZOOM sur \n Resultats L-V autour du point de depart :%s" % nb_individus_init)
    plt.axis([80,140,7,17])
    plt.show()


def distinction_Nt_Pt(y):
    """ A partir de y tel que les y[0] soient les proies\
    les y[1] les predateurs, renvoie le tableau\
    des proies, et celui des predateurs. """
    y_len = len(y)
    Nt = np.zeros(y_len)
    Pt = np.zeros(y_len)
    for i in range(y_len):
        Nt[i] = y[i][0]
        Pt[i] = y[i][1]
    return (Nt, Pt)    

def modele_proie_predateur():
    """modele_proie_predateurs calcule et affiche des resultats \
    issus du modele de Lotka-Volterra (modele proies/predateurs).\
    N(t) est le nombre de proies en fonction du temps.\
    P(t) est le nombre de predateurs en fonction du temps."""


    ## init : 
    ## ----
    
    t0 = 0.
    tf = 365.
    eps = 0.1
    
    #a : taux de reproduction des proies (independant des predateurs)
    #b : taux de mortalite des proies (du aux predateurs)
    #c : taux de reproduction des predateurs suite a la chasse de proies
    #d : taux de mortalite des predateurs (independant des proies)
    a = 0.25  # +25% proies / unite de temps
    b = 0.05  # -5% du nb de proies * predateurs
    c = 0.01  # +1% du nb de proies * predateurs
    d = 0.1   # -10% predateurs / unite de temps
    

    ##########################################
    ## 4. Resolution systeme proies/predateurs
    ##########################################
    
    nb_individus_init = np.array([80., 20.]) #([proies,predateurs])
   
    res_periodique = meth_epsilon(nb_individus_init, t0, tf, eps,\
                                  derivee_Nt_Pt(a, b, c, d), "rk4");
    (res_p_proies, res_p_pred) = distinction_Nt_Pt(res_periodique)
    #-> on separe le tableau des y[0] et des y[1] dans le resultat
    #pour pouvoir afficher

    plt.plot(x_values(t0, tf, len(res_periodique)), res_p_proies,"green",\
             label="N(t), nb proies")
    plt.plot(x_values(t0, tf, len(res_periodique)), res_p_pred,"purple",\
             label="P(t), nb predateurs")
    plt.legend()
    plt.xlabel("temps")
    plt.ylabel("nb d'individus")
    plt.title("Resultat periodique du modele Lotka-Volterra (en fonction du temps)")
    plt.axis([t0,tf,0,140])
    plt.show()

    ###############################################
    ## 5.  P(t) en fonction de N(t)
    ###############################################
    
    plt.plot(res_p_proies, res_p_pred,"purple")
    plt.xlabel("Nb proies N(t)")
    plt.ylabel("Nb predateurs P(t)")
    plt.title("Affichage de P(t) en fonction de N(t),\n \
    en utilisant le modele de Lotka-Volterra." )
    plt.show()

    ##############################################
    ## 6. Periode du resultat
    ##############################################

    print("Periode approximee du resultat de L-V: ",\
          round(approx_periode(res_p_proies, t0, tf),2))

    ################################################
    ## 7. les solutions autour d'un point de depart donne
    ################################################
    
    diagramme_solutions(nb_individus_init, 8, 2, \
                        a, b, c, d, t0, tf, eps)

    # ################################################
    # ## 8. Points singuliers
    # ################################################

    # ##Les deux points singuliers
    # singulier0 = np.array([0., 0.]) 
    # singulier_d_c_a_b = np.array([d/c, a/b])

    # ##Les resultats 
    # evol_pop_0 = meth_epsilon(singulier0, t0, tf, eps, \
    #                           derivee_Nt_Pt(a, b, c, d), "rk4"); 
    # evol_pop_2 = meth_epsilon(singulier_d_c_a_b, t0, tf, eps,\
    #                           derivee_Nt_Pt(a, b, c, d), "rk4");

    # ##Separation tableaux
    # (proie_evol_pop_0, pred_evol_pop_0) = distinction_Nt_Pt(evol_pop_0)
    # (proie_evol_pop_2, pred_evol_pop_2) = distinction_Nt_Pt(evol_pop_2)

    # ##Affichage
    # plt.plot(x_values(t0, tf, len(evol_pop_0)), proie_evol_pop_0,\
    #          "-p", label="N(t) (proies) - premier point fixe")
    # plt.plot(x_values(t0, tf, len(evol_pop_0)), pred_evol_pop_0,\
    #          "r", label="P(t) (predateurs) - premier point fixe")

    # plt.plot(x_values(t0, tf, len(evol_pop_2)), proie_evol_pop_2,\
    #          "-p", label="N(t) - second point fixe")
    # plt.plot(x_values(t0, tf, len(evol_pop_2)), pred_evol_pop_2,\
    #          "--g", label="P(t) - second point fixe")

    # plt.legend()
    # plt.xlabel("temps")
    # plt.ylabel("nb individus")
    # plt.title("Points singuliers - \n modele proies-predateurs de L-V")
    # plt.axis([t0,tf,-2,17])
    # plt.show()


def main():
    modele_malthusien()
    modele_de_verhulst()
    modele_proie_predateur()


if __name__== '__main__':
    main()
