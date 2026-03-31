import sys
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import sklearn

m_pca = 3
m_lda = 1

def load():
    filename = sys.argv[1]
    D = np.zeros((4, 150), dtype = np.float64)
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

def mcol(D):
    return D.reshape(-1, 1)

def mrow(D):
    return D.reshape(1, -1)

def normalization(D):
    
    mu = D.mean(1)
    mu= mcol(mu)
    DC = D-mu
    return DC, mu

def covariance(DC):
    return (DC@DC.T)/DC.shape[1]

def computevarstd(D):
    var = D.var(1)
    std = D.std(1)
    return var, std

def PCA(C, D, m):
    U, s, Vh = np.linalg.svd(C)

    # prendo le prime m colonne con varianza maggiore 
    P = U[:, 0:m]
    DP = P.T @ D

    return DP, P

def plot(DP, L):
    plt.figure()
    plt.scatter(DP[0, L==0], DP[1, L==0], label='Setosa')
    plt.scatter(DP[0, L==1], DP[1, L==1], label='Versicolor')
    plt.scatter(DP[0, L==2], DP[1, L==2], label='Virginica')
    plt.xlabel("Prima Direzione Principale")
    plt.ylabel("Seconda Direzione Principale")
    plt.title("Iris Dataset")
    plt.legend()
    plt.show()

def plot_histogram(DP, L):
    plt.figure()
    plt.hist(DP[0, L==1], bins=15, alpha=0.5, label='Versicolor', edgecolor='black')
    plt.hist(DP[0, L==2], bins=15, alpha=0.5, label='Virginica', edgecolor='black')
    plt.xlabel("Direzione LDA (1D)")
    plt.title("Istogramma della Proiezione LDA")
    plt.legend()
    plt.show()

def LDA(D, L, m):

    SW = np.zeros((D.shape[0], D.shape[0]), dtype = np.float64)
    SB = np.zeros((D.shape[0], D.shape[0]), dtype = np.float64)
    _ , mu = normalization(D)

    for c in np.unique(L):
        
        DCc, muc = normalization(D[:, L==c])
        SWc = covariance(DCc)
        SW += DCc.shape[1] * SWc
        SB += DCc.shape[1]*((muc-mu)@(muc-mu).T)

    SW = SW / D.shape[1]
    SB = SB / D.shape[1]
    
    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]
    DP = W.T @ D
    return DP, W

def load_iris():
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    
def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

#L, D = load()
DIris, LIris = load_iris()
D = DIris[:, LIris != 0]
L = LIris[LIris != 0]

(DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
DC, mu = normalization(DTR)
covarianceMatrix = covariance(DC)

DP_PCA, P = PCA(covarianceMatrix, DTR, m_pca)
#plot(DP_PCA, L)
#DP_PCA = -DP_PCA
#P = -P

DP_LDA, W = LDA(DP_PCA, LTR, m_lda)
DP_LDA = -DP_LDA
W = -W

#plot_histogram(DP_PCA, LTR)
plot_histogram(DP_LDA, LTR)

#threshold = (DP_PCA[0, LTR==1].mean() + DP_PCA[0, LTR==2].mean()) / 2.0
threshold = (DP_LDA[0, LTR==1].mean() + DP_LDA[0, LTR==2].mean()) / 2.0

DVAL_PCA = P.T @ DVAL
DVAL_LDA = W.T @ DVAL_PCA

PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
PVAL[DVAL_LDA[0] >= threshold] = 2
PVAL[DVAL_LDA[0] < threshold] = 1

#PVAL[DVAL_PCA[0] >= threshold] = 2
#PVAL[DVAL_PCA[0] < threshold] = 1

nError = np.sum(PVAL != LVAL)
nElement = LVAL.size

print(f"{nError} / {nElement}")