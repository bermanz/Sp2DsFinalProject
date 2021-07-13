import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas
import pandas as pd
from sklearn.datasets import make_spd_matrix
import scipy.stats as stats
font = {'size': 16}
matplotlib.rc('font', **font)

def getMogData(K:int=5, sigma:float=5, nDims:int=2 ,nSamples:int=100, debug:bool=True):
    """Generates a MOG model and draws nSamples from it"""
    mu_k = np.random.multivariate_normal(mean=np.zeros(shape=nDims), cov=sigma**2 * np.identity(nDims), size=K)
    c = np.random.choice(range(K), size=nSamples, p=np.ones(shape=K) / K)
    mu_c = mu_k[c]
    # cov_k = np.zeros((K, nDims, nDims))
    # for k in range(K):
    #     cov_k[k] = make_spd_matrix(n_dim=nDims)
    cov_k = np.ones((K, nDims, nDims))

    x_mu_c = lambda i: np.random.multivariate_normal(mean=mu_c[i].squeeze(axis=0), cov=cov_k[c[i]].squeeze(axis=0))
    x = np.apply_along_axis(x_mu_c, axis=1, arr=np.expand_dims(np.arange(nSamples), axis=1))# np.random.multivariate_normal(mu_c, cov=np.identity(nDims))
    x_df = pd.DataFrame(x.tolist(), columns=["x"])
    x_df["c"] = c
    x_df["mu"] = mu_c.tolist()
    x_df["cov"] = cov_k[c].tolist()
    if debug:
        fig, ax = plt.subplots(2, 1)
        # plt.axis("equal")
        ax[0].hist(c)
        ax[0].set_xlabel("k")
        ax[0].set_ylabel("nAssignments")
        ax[0].set_title("Cluster Assignments Balance")

        # ax[1].axis("equal")
        for k in range(K):
            k_data = x_df.loc[x_df.c == k]
            mean = k_data.mu.iloc[0][0]
            var = k_data.loc[:, "cov"].iloc[0][0][0]
            plotGauusian(mean, var, f"k={k}", ax[1])
            ax[1].scatter(k_data.x, stats.norm.pdf(k_data.x, loc=mean, scale=np.sqrt(var)))
            # x_span = np.linspace(mean - 3*var, mean + 3*var, 100)
            # ax[1].plot(x_span, stats.norm.pdf(x_span, mean, var), label=f"k={k}")
            # ax[1].scatter(k_data.x, stats.norm.pdf(k_data.x, mean, var))


        # ax[1].axes.set_aspect('equal')
        ax[1].legend()
        ax[1].set_title("Sampled Data")
        ax[1].set_xlabel("$x$")
        ax[1].set_ylabel("$p(x|c, \mu)$")
        # ax[1].set_ylim((0, 0.5))
        ax[1].grid()

    return x_df

def plotGauusian(mean, var, label, ax):
    std = np.sqrt(var)
    x_span = np.linspace(mean - 3 * std, mean + 3 * std, 100)
    ax.plot(x_span, stats.norm.pdf(x_span, loc=mean, scale=std), label=label)



def CAVI(data:pd.DataFrame, sigma:float):
    """Coordinate Ascent Variational Inference for GMM"""

    n = len(data)
    K = len(data.c.unique())

    # Initialization:
    m = np.random.normal(size=(1, K))
    s2 = np.random.normal(size=(1, K))
    phi = np.ones(shape=(n, K)) / K

    x = np.expand_dims(data.x.to_numpy(), axis=1)  # set data as (n,1) ndarray

    # implementation:
    is_conv = False
    conv_tol = 1e-10
    elbo = []

    while not is_conv:
        # update variational assignments:
        phi = np.exp(x @ m - 0.5 * (s2 + m**2))
        phi /= phi.sum(axis=1, keepdims=True)

        # update variational densities:
        den = 1/sigma**2 + phi.sum(axis=0, keepdims=True)
        m = (phi * x).sum(axis=0, keepdims=True) / den
        s2 = 1 / den

        elbo.append(calcElbo(x, sigma, np.expand_dims(m, axis=0), np.expand_dims(s2, axis=0), phi))
        if len(elbo) > 2 and np.abs(elbo[-2] - elbo[-1]) < conv_tol:
            is_conv = True

    return phi, m, s2, elbo

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
    elbo_terms[0] = -1/(2*sigma**2) * (s2_hat + m_hat ** 2).sum()
    elbo_terms[1] = (phi_hat * (x @ m_hat - 0.5*(s2_hat + m_hat ** 2))).sum()
    elbo_terms[2] = -(phi_hat * np.log(phi_hat)).sum()
    elbo_terms[3] = 0.5*np.log(s2_hat).sum()

    return elbo_terms.sum()

if __name__ == "__main__":
    sigma = 5
    K = 5
    data = getMogData(K=K, sigma=sigma, nDims=1)
    phi, m, s2, elbo = CAVI(data, sigma)
    fig, ax = plt.subplots()
    for k in range(K):
        k_data = data.loc[data.c == k]
        mean = k_data.mu.iloc[0][0]
        var = k_data.loc[:, "cov"].iloc[0][0][0]
        ax.scatter(k_data.x, stats.norm.pdf(k_data.x, loc=mean, scale=np.sqrt(var)), label=f"k={k} data")
        mean_hat = m.squeeze()[k]
        var_hat = s2.squeeze()[k]
        # plotGauusian(mean_hat, var_hat, f"k={k} variational", ax)
        plotGauusian(np.random.normal(loc=mean_hat, scale=var_hat), 1, f"k={k} variational", ax)
