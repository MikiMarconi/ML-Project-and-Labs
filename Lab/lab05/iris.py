import sys
import numpy as np
import scipy
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

def plotDistribution():
    plt.figure()
    XPlot = np.linspace(-8, 12, 1000)
    m = np.ones((1,1)) * 1.0
    C = np.ones((1,1)) * 2.0
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(mrow(XPlot), m, C)))
    plt.show()

def loglikelihood(X, mu, C):
    Y = logpdf_GAU_ND(X, mu, C)
    return np.sum(Y)

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

def computeScoreDatasetMVG(DTR, LTR, DTE):

    DTR0 = DTR[:, LTR == 0]
    DTR1 = DTR[:, LTR == 1]
    DTR2 = DTR[:, LTR == 2]

    DC0, mu0 = normalization(DTR0)
    DC1, mu1 = normalization(DTR1)
    DC2, mu2 = normalization(DTR2)

    CM0 = covariance(DC0)
    CM1= covariance(DC1)
    CM2 = covariance(DC2)

    S = np.zeros((3, 50), dtype= np.float64)
    S[0, :] = logpdf_GAU_ND(DTE, mu0, CM0)
    S[1, :] = logpdf_GAU_ND(DTE, mu1, CM1)
    S[2, :] = logpdf_GAU_ND(DTE, mu2, CM2)

    return S

def computeScoreDatasetNBG(DTR, LTR, DTE):

    DTR0 = DTR[:, LTR == 0]
    DTR1 = DTR[:, LTR == 1]
    DTR2 = DTR[:, LTR == 2]

    DC0, mu0 = normalization(DTR0)
    DC1, mu1 = normalization(DTR1)
    DC2, mu2 = normalization(DTR2)

    CM0 = covariance(DC0) * np.eye(DC0.shape[0])
    CM1= covariance(DC1) * np.eye(DC1.shape[0])
    CM2 = covariance(DC2) * np.eye(DC2.shape[0])

    S = np.zeros((3, 50), dtype= np.float64)
    S[0, :] = logpdf_GAU_ND(DTE, mu0, CM0)
    S[1, :] = logpdf_GAU_ND(DTE, mu1, CM1)
    S[2, :] = logpdf_GAU_ND(DTE, mu2, CM2)

    return S

def MVG_NBG_TCG(S, LTE):
    priorProbability = 1.0/3.0
    logSJoint = S + mcol(np.log(priorProbability))
    logSMarginal = mrow(scipy.special.logsumexp(logSJoint, axis = 0))

    logSPost = logSJoint - logSMarginal
    SPost = np.exp(logSPost)
    predictedLabels = np.argmax(SPost, axis=0)
    nCorrect = np.sum(predictedLabels == LTE)

    accuracy = nCorrect / LTE.size
    errorRate = 1-accuracy
    return errorRate

def TCG(DTR, LTR, DTE):
    DTR0 = DTR[:, LTR == 0]
    DTR1 = DTR[:, LTR == 1]
    DTR2 = DTR[:, LTR == 2]

    DC0, mu0 = normalization(DTR0)
    DC1, mu1 = normalization(DTR1)
    DC2, mu2 = normalization(DTR2)

    CM0 = covariance(DC0)
    CM1= covariance(DC1)
    CM2 = covariance(DC2)
    CM_tied  = (1/DTR.shape[1]) * ((DTR0.shape[1] * CM0) + (DTR1.shape[1] * CM1) + (DTR2.shape[1] * CM2))
    
    S = np.zeros((3, 50), dtype= np.float64)
    S[0, :] = logpdf_GAU_ND(DTE, mu0, CM_tied)
    S[1, :] = logpdf_GAU_ND(DTE, mu1, CM_tied)
    S[2, :] = logpdf_GAU_ND(DTE, mu2, CM_tied)
    
    return S

def binaryClassifierTCG(DTR, LTR, DTE, LTE):
    DTR1 = DTR[:, LTR == 1]
    DTR2 = DTR[:, LTR == 2]

    DC1, mu1 = normalization(DTR1)
    DC2, mu2 = normalization(DTR2)

    CM1= covariance(DC1)
    CM2 = covariance(DC2)
    CM_tied  = (1/DTR.shape[1]) * ((DTR1.shape[1] * CM1) + (DTR2.shape[1] * CM2))
    
    Y1 = logpdf_GAU_ND(DTE, mu1, CM_tied)
    Y2 = logpdf_GAU_ND(DTE, mu2, CM_tied)
    LLR = Y2 - Y1

    predict = np.zeros((LLR.shape[0]), dtype = np.int32)

    for i in range(LLR.shape[0]):
        if LLR[i] >= 0:
            predict[i] = 2
        else:
            predict[i] = 1

    nCorrect = np.sum(predict == LTE)
    accuracy = nCorrect / LTE.size
    errorRate = 1-accuracy
    
    return errorRate, predict

def binaryClassifierMVG(DTR, LTR, DTE, LTE):
    DTRBinary1 = DTR[:, LTR == 1]
    DTRBinary2 = DTR[:, LTR == 2]

    DB1, mu1 = normalization(DTRBinary1)
    DB2, mu2 = normalization(DTRBinary2)

    CB1 = covariance(DB1)
    CB2 = covariance(DB2)

    Y1 = logpdf_GAU_ND(DTE, mu1, CB1)
    Y2 = logpdf_GAU_ND(DTE, mu2, CB2)
    LLR = Y2 - Y1

    predict = np.zeros((LLR.shape[0]), dtype = np.int32)

    for i in range(LLR.shape[0]):
        if LLR[i] >= 0:
            predict[i] = 2
        else:
            predict[i] = 1

    nCorrect = np.sum(predict == LTE)
    accuracy = nCorrect / LTE.size
    errorRate = 1-accuracy
    
    return errorRate

def checkBinaryClassificationEqualityTCG_LDA(DTR_bin, LTR_bin, DTE_bin, LTE_bin, predictTCG):
    DP_LDA, W = LDA(DTR_bin, LTR_bin, m_lda)
    mu1_LDA = DP_LDA[0, LTR_bin==1].mean()
    mu2_LDA = DP_LDA[0, LTR_bin==2].mean()
    threshold = (mu1_LDA + mu2_LDA) / 2.0
    DVAL_LDA = W.T @ DTE_bin

    PVAL = np.zeros(shape=LTE_bin.shape, dtype=np.int32)
    if mu2_LDA > mu1_LDA:
        PVAL[DVAL_LDA[0] >= threshold] = 2
        PVAL[DVAL_LDA[0] < threshold] = 1
    else:
        PVAL[DVAL_LDA[0] < threshold] = 2
        PVAL[DVAL_LDA[0] >= threshold] = 1

    checkUgual = np.all(PVAL == predictTCG)
    
    return checkUgual

DIris, LIris = load_iris()
(DTR, LTR), (DTE, LTE) = split_db_2to1(DIris, LIris)

SMVG = computeScoreDatasetMVG(DTR, LTR, DTE)
errorRateMVG = MVG_NBG_TCG(SMVG, LTE)
print(f"Error rate MVG: {errorRateMVG:.3f}")

SNGB = computeScoreDatasetNBG(DTR, LTR, DTE)
errorRateNBG = MVG_NBG_TCG(SNGB, LTE)
print(f"Error rate NBG: {errorRateNBG:.3f}")

STCG = TCG(DTR, LTR, DTE)
errorRateTCG = MVG_NBG_TCG(STCG, LTE)
print(f"Error rate TCG: {errorRateTCG:.3f}")

DBin_full = DIris[:, LIris != 0]
LBin_full = LIris[LIris != 0]

(DTR_bin, LTR_bin), (DTE_bin, LTE_bin) = split_db_2to1(DBin_full, LBin_full)

errorRateBinaryClassifierMVG = binaryClassifierMVG(DTR_bin, LTR_bin, DTE_bin, LTE_bin)
print(f"Error rate Binary Classifier with MVG: {errorRateBinaryClassifierMVG}")

errorRateBinaryClassifierTCG, predictTCG = binaryClassifierTCG(DTR_bin, LTR_bin, DTE_bin, LTE_bin)
print(f"Error rate Binary Classifier with TCG: {errorRateBinaryClassifierTCG}")

check = checkBinaryClassificationEqualityTCG_LDA(DTR_bin, LTR_bin, DTE_bin, LTE_bin, predictTCG)
print(f"Check if LDA and TCG obtain the same prediction results: {check}")