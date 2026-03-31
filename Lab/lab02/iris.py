import sys
import numpy as np
import matplotlib.pyplot as plt

def load():
    filename = sys.argv[1]
    D = np.zeros((4, 150), dtype = np.float32)
    L = np.zeros(150, dtype = np.int32)

    with open(filename, 'r') as f:

        count = 0
        
        for line in f:
            datasetElement = line.strip().split(',')
            D[0, count] = float(datasetElement[0])
            D[1, count] = float(datasetElement[1])
            D[2, count] = float(datasetElement[2])
            D[3, count] = float(datasetElement[3])
            
            if datasetElement[4] == "Iris-setosa":
                L[count] = 0
            
            elif datasetElement[4] == "Iris-versicolor":
                L[count] = 1

            elif datasetElement[4] == "Iris-virginica":
                L[count] = 2
                
            count += 1
    return L, D

def plot(D, L):
    
    M0 = (L == 0)
    M1 = (L == 1)
    M2 = (L == 2)

    D0 = D[:, M0]
    D1 = D[:, M1]
    D2 = D[:, M2]
    
    plothist(D0, D1, D2)
    plotscatter(D0, D1, D2)
    return D0, D1, D2


def plothist(D0, D1, D2):
    for idx in range(4):

        plt.figure()
        plt.hist(D0[idx, :], bins=10, density=True, alpha=0.4, label='Setosa')
        plt.hist(D1[idx, :], bins=10, density=True, alpha=0.4, label='Versicolor')
        plt.hist(D2[idx, :], bins=10, density=True, alpha=0.4, label='Virginica')

        if idx == 0:
            plt.title("Sepal Length")
        elif idx == 1:
            plt.title("Sepal Width")
        elif idx == 2:
            plt.title("Petal Length") 
        elif idx == 3:
            plt.title("Petal Width")
        
        plt.xlabel("cm")
        plt.ylabel("Density")
        plt.legend()
    plt.show()

def plotscatter(D0, D1, D2):

    element = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
    for idx in range(4):
        for idy in range(4):

            if idx < idy:
                plt.figure()
                plt.scatter(D0[idx, :], D0[idy, :], label='Setosa')
                plt.scatter(D1[idx, :], D1[idy, :], label='Versicolor')
                plt.scatter(D2[idx, :], D2[idy, :], label='Virginica')
                plt.xlabel(element[idx])
                plt.ylabel(element[idy])
                plt.legend()

    plt.show()

def mcol(D):
    return D.reshape(-1, 1)

def mrow(D):
    return D.reshape(1, -1)

def normalization(D):
    
    mu = D.mean(1)
    mu= mcol(mu)
    DC = D-mu
    return DC

def covariance(DC):
    return (DC@DC.T)/D.shape[1]

def computevarstd(D):
    var = D.var(1)
    std = D.std(1)
    return var, std

def print_statistics(D0, D1, D2):
    classi = [("Setosa", D0), ("Versicolor", D1), ("Virginica", D2)]
    nomi_misure = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
    
    for nome_classe, matrice_dati in classi:
        print(f"========================================")
        print(f" CLASSE: {nome_classe.upper()}")
        print(f"========================================")
        medie = matrice_dati.mean(1)
        varianze = matrice_dati.var(1)
        dev_standard = matrice_dati.std(1)
        
        for i in range(4):
            print(f"- {nomi_misure[i]}:")
            print(f"    Media: {medie[i]:.4f}")
            print(f"    Varianza: {varianze[i]:.4f}")
            print(f"    Dev. Standard: {dev_standard[i]:.4f}")
        print("\n") 


L, D = load()
D0, D1, D2 =plot(D, L)
DC = normalization(D)
plot(DC, L)

covarianceMatrix = covariance(DC)
print_statistics(D0, D1, D2)
