import matplotlib.pyplot as plt
import numpy as np

def show_vector_field(f, x0, y0, xmax, ymax, dx, dy):
    """ Dessine le champ des tangentes de l'equation differentielle 
    donnee par le probleme de Cauchy ((x0,y0), f). 
    Uniquement pour les fonctions de dimension 2.
    Le decoupage se fait entre les points (x0,y0) et (xmax, ymax),
    avec des segments de longueur resp. dx et dy."""

    #Initialisation
    X = np.zeros([len(np.arange(x0, xmax, dx)), 
                  len(np.arange(y0, ymax, dy))])
    Y = np.copy(X)
    U = np.copy(X)
    V = np.copy(Y)

    #Remplissage de X, Y, U et V
    for i in range(X.shape[0]): #lignes
        for j in range(X.shape[1]): #colonnes
            X[i,j] = x0 + i*dx
            Y[i,j] = y0 + j*dy
            
            derivee = f([X[i,j],Y[i,j]], 0)
            U[i,j] = derivee[0]
            V[i,j] = derivee[1]

    #Affichage
    plt.figure()
    plt.quiver(X,Y,U,V)
    plt.title("Champ des tangentes pour le probleme de Cauchy (f,y0)")
    plt.show()
            
def fex(y,t):
    return 2*y

show_vector_field(fex, 0,1, 100,200, 1,4)

