import numpy as np
from Plot import plothist, plot_features_gaussian
import Logic
from Stats import print_statistics


L, D = Logic.load()
(DTR, LTR), (DVAL, LVAL) = Logic.split_db_2to1(D, L)
DC, mu = Logic.normalization(DTR)
covarianceMatrix = Logic.covariance(DC)

DP_PCA, P = Logic.PCA(covarianceMatrix, DTR, 4)
#plothist(DP_PCA, LTR)

DP_LDA, W = Logic.LDA(DP_PCA, LTR, 1)
#plothist(DP_LDA, LTR)

threshold = (DP_LDA[0, LTR==0].mean() + DP_LDA[0, LTR==1].mean()) / 2.0

DVAL_PCA = P.T @ DVAL
DVAL_LDA = W.T @ DVAL_PCA

PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
PVAL[DVAL_LDA[0] >= threshold] = 1
PVAL[DVAL_LDA[0] < threshold] = 0

nError = np.sum(PVAL != LVAL)
nElement = LVAL.size
#print(f"{nError} / {nElement}")

plot_features_gaussian(DTR, LTR)
