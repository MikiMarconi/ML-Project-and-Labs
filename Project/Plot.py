import matplotlib.pyplot as plt
import numpy as np
import Logic

def plothist(DP, L):
    plt.figure()
    plt.hist(DP[0, L==0], bins=15, alpha=0.5, label='False', edgecolor='black')
    plt.hist(DP[0, L==1], bins=15, alpha=0.5, label='True', edgecolor='black')
    plt.legend()
    plt.show()

def plot_features_gaussian(X, L):
    X0 = X[:, L == 0]
    X1 = X[:, L == 1]

    for i in range(X.shape[0]):
        plt.figure()
        xi0 = X0[i:i+1, :]
        xi1 = X1[i:i+1, :]

        _, mu0 = Logic.normalization(xi0)
        _, mu1 = Logic.normalization(xi1)
        C0 = Logic.covariance(xi0)
        C1 = Logic.covariance(xi1)

        plt.hist(xi0.ravel(), bins=50, density=True, alpha=0.5, label='False (0)', edgecolor='black')
        plt.hist(xi1.ravel(), bins=50, density=True, alpha=0.5, label='True (1)', edgecolor='black')
        

        XPlot = np.linspace(X[i, :].min(), X[i, :].max(), 1000)
        XPlot_2D = Logic.mrow(XPlot)
        pdf0 = np.exp(Logic.logpdf_GAU_ND(XPlot_2D, mu0, C0))
        pdf1 = np.exp(Logic.logpdf_GAU_ND(XPlot_2D, mu1, C1))

        plt.plot(XPlot.ravel(), pdf0, color='blue', linewidth=2, label='Gaussian False')
        plt.plot(XPlot.ravel(), pdf1, color='orange', linewidth=2, label='Gaussian True')
        
        plt.title(f"Feature {i}")
        plt.legend()
    plt.show()
