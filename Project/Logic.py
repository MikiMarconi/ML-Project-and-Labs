import sys
import numpy as np
import scipy

def load():
    filename = sys.argv[1]
    listElement = []
    listClass = []

    with open(filename, 'r') as f:
        
        for line in f:
            feature = []
            feature.append(float(line.strip().split(',')[0]))
            feature.append(float(line.strip().split(',')[1]))
            feature.append(float(line.strip().split(',')[2]))
            feature.append(float(line.strip().split(',')[3]))
            feature.append(float(line.strip().split(',')[4]))
            feature.append(float(line.strip().split(',')[5]))
            listClass.append(int(line.strip().split(',')[6]))
            listElement.append(feature)

        # D = (6, NElement); L = (NElement)
        D = np.array(listElement).T
        L = np.array(listClass)

    return L, D

def computevarstd(D):
    var = D.var(1)
    std = D.std(1)
    return var, std

def covariance(DC):
    return (DC@DC.T)/DC.shape[1]

def mcol(D):
    return D.reshape(-1, 1)

def mrow(D):
    return D.reshape(1, -1)

def normalization(D):
    
    mu = D.mean(1)
    mu= mcol(mu)
    DC = D-mu
    return DC, mu

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

    if DP[0, L==0].mean() > DP[0, L==1].mean():
        DP = -DP
        W = -W
        
    return DP, W

def PCA(C, D, m):
    U, s, Vh = np.linalg.svd(C)

    # prendo le prime m colonne con varianza maggiore 
    P = U[:, 0:m]
    DP = P.T @ D

    return DP, P

def logpdf_GAU_ND(X, mu, C):
    M = X.shape[0]
    Y = np.zeros((X.shape[1]), dtype = np.float64)
    _, log_det = np.linalg.slogdet(C)
    log_det = log_det * 0.5
    constNorm = -((M/2) * np.log(2 * np.pi))
    invC = np.linalg.inv(C)

    for i in range(X.shape[1]):
        xi = X[:, i:i+1]
        Y[i] = (constNorm - log_det - (0.5 * (xi - mu).T @ invC @ (xi - mu)).item())

    return Y

def loglikelihood(X, mu, C):
    Y = logpdf_GAU_ND(X, mu, C)
    return np.sum(Y)

def binaryClassifierMVG(DTR, LTR, DTE):
    DTRBinary0 = DTR[:, LTR == 0]
    DTRBinary1 = DTR[:, LTR == 1]

    DB0, mu0 = normalization(DTRBinary0)
    DB1, mu1 = normalization(DTRBinary1)

    CB0= covariance(DB0)
    CB1 = covariance(DB1)

    Y0 = logpdf_GAU_ND(DTE, mu0, CB0)
    Y1 = logpdf_GAU_ND(DTE, mu1, CB1)
    LLR = Y1 - Y0

    return LLR, CB0, CB1

def binaryClassifierTCG(DTR, LTR, DTE, LTE):
    DTR0 = DTR[:, LTR == 0]
    DTR1 = DTR[:, LTR == 1]

    DC0, mu0 = normalization(DTR0)
    DC1, mu1 = normalization(DTR1)

    CM0= covariance(DC0)
    CM1 = covariance(DC1)
    CM_tied  = (1/DTR.shape[1]) * ((DTR0.shape[1] * CM0) + (DTR1.shape[1] * CM1))
    
    Y0 = logpdf_GAU_ND(DTE, mu0, CM_tied)
    Y1 = logpdf_GAU_ND(DTE, mu1, CM_tied)
    LLR = Y1 - Y0
    return LLR

def binaryClassifierNBG(DTR, LTR, DTE):

    DTR0 = DTR[:, LTR == 0]
    DTR1 = DTR[:, LTR == 1]

    DC0, mu0 = normalization(DTR0)
    DC1, mu1 = normalization(DTR1)

    CM0 = covariance(DC0) * np.eye(DC0.shape[0])
    CM1= covariance(DC1) * np.eye(DC1.shape[0])

    Y0 = logpdf_GAU_ND(DTE, mu0, CM0)
    Y1 = logpdf_GAU_ND(DTE, mu1, CM1)
    LLR = Y1 - Y0
    return LLR

def applyThreshold(LLR):
    predict = np.zeros((LLR.shape[0]), dtype = np.int32)

    for i in range(LLR.shape[0]):
        if LLR[i] >= 0:
            predict[i] = 1
        else:
            predict[i] = 0

    return predict

def computeError(predict, LTE):
    nCorrect = np.sum(predict == LTE)
    accuracy = nCorrect / LTE.size
    errorRate = 1-accuracy
    
    return errorRate