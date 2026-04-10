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
DTR_PCA = P.T @ DTR
DVAL_PCA = P.T @ DVAL

DVAL_LDA = W.T @ DVAL_PCA

PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
PVAL[DVAL_LDA[0] >= threshold] = 1
PVAL[DVAL_LDA[0] < threshold] = 0

nError = np.sum(PVAL != LVAL)
nElement = LVAL.size
print(f"LDA with PCA: {(nError / nElement)*100:3f}%")
#plot_features_gaussian(DTR, LTR)

LLRMVG, CB0, CB1 = Logic.binaryClassifierMVG(DTR_PCA, LTR, DVAL_PCA)
LLRTCG = Logic.binaryClassifierTCG(DTR_PCA, LTR, DVAL_PCA, LVAL)
LLRNBG = Logic.binaryClassifierNBG(DTR_PCA, LTR, DVAL_PCA)

predictMVG = Logic.applyThreshold(LLRMVG)
predictTCG = Logic.applyThreshold(LLRTCG)
predictNBG = Logic.applyThreshold(LLRNBG)

errorRateMVG = Logic.computeError(predictMVG, LVAL)
errorRateTCG = Logic.computeError(predictTCG, LVAL)
errorRateNBG = Logic.computeError(predictNBG, LVAL)

print(f"MVG: {(errorRateMVG*100):1f}%")
print(f"TCG: {(errorRateTCG*100):1f}%")
print(f"NBG: {(errorRateNBG*100):1f}%")

Corr0 = CB0 / ( Logic.mcol(CB0.diagonal()**0.5) * Logic.mrow(CB0.diagonal()**0.5) )
Corr1 = CB1 / ( Logic.mcol(CB1.diagonal()**0.5) * Logic.mrow(CB1.diagonal()**0.5) )
