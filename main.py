import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas
import pandas as pd
from sklearn.datasets import make_spd_matrix
import scipy.stats as stats

font = {'size': 16}
matplotlib.rc('font', **font)
matplotlib.rc('figure', **{"figsize": (19.2, 9.77), "autolayout": True})
#
import seaborn as sns
from utils import confidence_ellipse, plotData, plotDist, to_numpy, plotCaviState, genResDir, genGifFromPlots, \
    getMogData, \
    plotCaviIter, makeGif
from pathlib import Path


def CAVI(data: pd.DataFrame, sigma:float=1, conv_tol=1e-5, max_iter=100):
    """Coordinate Ascent Variational Inference for GMM"""

    n = len(data)
    K = len(data.c.unique())
    dims = len(data.mu[0])

    # Initialization:
    m = np.random.normal(size=(dims, K))
    s2 = np.ones(shape=(dims, K))
    phi = np.ones(shape=(n, K)) / K

    debug_data = pd.DataFrame(index=range(max_iter), columns=["m", "s2", "phi", "elbo"])
    debug_data.loc[0, "m"] = m
    debug_data.loc[0, "s2"] = s2
    debug_data.loc[0, "phi"] = phi
    debug_data.loc[0, "elbo"] = None

    x = to_numpy(data.x)  # set data as (n,1) ndarray

    # implementation:
    is_conv = False
    iter = 0
    elbo_prev = np.inf
    while not is_conv and iter < max_iter:

        # update variational assignments:
        phi = getLocalVar(x, m, s2)

        # update variational densities:
        m, s2 = getGlobalVar(x, phi, sigma)

        # calculate ELBO:
        elbo = calcElbo(x, sigma, m, s2, phi)

        # assert convergence:
        if np.abs(elbo_prev - elbo) < conv_tol:
            is_conv = True

        elbo_prev = elbo
        iter += 1

        # assign debug variables:
        debug_data.loc[iter, "m"] = m
        debug_data.loc[iter, "s2"] = s2
        debug_data.loc[iter, "phi"] = phi
        debug_data.loc[iter, "elbo"] = elbo

    debug_data.dropna(inplace=True, how="all")
    return phi, m, s2, debug_data

def getLocalVar(x, m, s2):
    """Calculate the local latent variational factors (Phi)"""
    phi = np.exp(x @ m - 0.5 * (s2 + m ** 2).sum(axis=0))
    phi /= phi.sum(axis=1, keepdims=True)
    return phi

def getGlobalVar(x, phi, sigma):
    """Calculate the global latent variational factors (m, s^2)"""
    # update variational densities:
    den = 1 / sigma ** 2 + phi.sum(axis=0, keepdims=True)
    m = (x.T @ phi) / den
    s2 = 1 / den
    return m, s2

def calcElbo(x: np.ndarray, sigma: float, m_hat: np.ndarray, s2_hat: np.ndarray, phi_hat: np.ndarray):
    """Calculates the evidence lower bound given the estimates of the latent variable's distributions

        Parameters:
            x - (n,1) array of the sampled data
            sigma - the standard deviation of the different Gaussians generator (hyper-parameter)
            m_hat - (1,K) means array of the variational Gaussians factors
            s2_hat - (1,K) variance array variational Gaussians factors
            phi_hat - (n,K) variational assignments matrix
    """
    elbo_terms = np.zeros(4)  # prepare a vector for the different elbo terms
    elbo_terms[0] = -1 / (2 * sigma ** 2) * (s2_hat + (m_hat ** 2).sum(axis=0)).sum()
    elbo_terms[1] = (phi_hat * (x @ m_hat - 0.5 * (s2_hat + (m_hat ** 2)).sum(axis=0))).sum()
    elbo_terms[2] = -(phi_hat * np.log(phi_hat)).sum()
    elbo_terms[3] = -0.5 * np.log(2 * s2_hat).sum()

    return elbo_terms.sum()


if __name__ == "__main__":
    from matplotlib.animation import FuncAnimation
    sigma = 3
    K = 5
    samples = getMogData(K=K, sigma=sigma, nSamples=int(1e3), debug=False)

    # inspect ELBOs sensitivity to initialization:
    phi, m, s2, debug_data = CAVI(samples)
    makeGif(samples, debug_data)
