import matplotlib.pyplot as plt
import numpy as np

def show_vector_field(f, x0, y0, xmax, ymax, dx, dy):
    """ Dessine le champ des tangentes de l'equation differentielle 
    donnee par le probleme de Cauchy (y0, f). 
    Uniquement pour les fonctions de dimension 2"""

    #Initialisation
    X = np.zeros([len(np.arange(x0, xmax, dx)), 
                  len(np.arange(y0, ymax, dy))])
    Y = np.copy(X)
    U = np.copy(X)
    V = np.copy(Y)

    n = X.shape[0] #lignes
    m = X.shape[1] #colonnes
    
    #Remplissage de X et Y
    X[0,:] = x0
    Y[:,0] = y0
    for i in range(1,n):
        X[i,:] = x0 + i*dx
    for j in range(1,m):
        Y[:,j] = y0 + j*dy
        
    #Remplissage de U et V
    for i in range(n):
        for j in range(m):
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

