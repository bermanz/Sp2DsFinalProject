import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas
import pandas as pd
from sklearn.datasets import make_spd_matrix
import scipy.stats as stats
font = {'size': 16}
matplotlib.rc('font', **font)
import seaborn as sns
from utils import confidence_ellipse

def getMogData(K:int=5, sigma:float=1, nDims:int=2 ,nSamples:int=100, debug:bool=True):
    """Generates a MOG model and draws nSamples from it"""
    mu_k = np.random.multivariate_normal(mean=np.zeros(shape=nDims), cov=sigma**2 * np.identity(nDims), size=K)
    c = np.random.choice(range(K), size=nSamples, p=np.ones(shape=K) / K)
    mu_c = mu_k[c]
    cov_k = np.zeros((K, nDims, nDims))
    for k in range(K):
    #     randSquaredMat = np.random.normal(size=(nDims, nDims))
    #     cov_k[k] = randSquaredMat.T @ randSquaredMat
    #     # cov_k[k] = make_spd_matrix(n_dim=nDims)
        cov_k[k] = np.identity(nDims)

    x_mu_c = lambda i: np.random.multivariate_normal(mean=mu_c[i].squeeze(axis=0), cov=cov_k[c[i]].squeeze(axis=0))
    x = np.apply_along_axis(x_mu_c, axis=1, arr=np.expand_dims(np.arange(nSamples), axis=1))# np.random.multivariate_normal(mu_c, cov=np.identity(nDims))
    x_df = pd.DataFrame(index=np.arange(nSamples))
    x_df["x"] = x.tolist()
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
            mean = np.array(k_data.mu.iloc[0])
            cov = np.array(k_data.loc[:, "cov"].iloc[0])
            # plotGauusian(mean, cov, f"k={k}", ax[1])
            getCoord = lambda data, i: np.array([lst[i] for lst in data])
            ax[1].plot(getCoord(k_data.x, 0), getCoord(k_data.x, 1), 'o')
            confidence_ellipse(getCoord(k_data.x, 0), getCoord(k_data.x, 1), ax=ax[1],
                               facecolor=ax[1].lines[-1].get_color(), alpha=0.5)
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

def plotGauusian(mean, cov, label, ax, *args):
    dist = stats.multivariate_normal(mean=mean, cov=cov)
    std = np.sqrt(cov)
    plotSpan = lambda i: np.linspace(mean[i] - 3 * std.diagonal()[i], mean[i] + 3 * std.diagonal()[i], 100)
    if len(mean) > 1:
        diag = std.diagonal()
        x, y = plotSpan(0), plotSpan(1)
        confidence_ellipse(x, y, ax=ax)
        ax.pcolormesh(x.flatten().T, y.flatten().T, dist.pdf(plot_span.T), shading='auto', cmap=plt.cm.BuGn_r, label=label, *args)
    else:
        plot_span = plotSpan(std)
        ax.plot(plot_span, dist.pdf(plot_span), label=label, *args)


def to_numpy(seriesOfLists):
    listOfLists = [x for x in seriesOfLists]
    return np.array(listOfLists)

def CAVI(data:pd.DataFrame, sigma:float, debug=True):
    """Coordinate Ascent Variational Inference for GMM"""

    n = len(data)
    K = len(data.c.unique())
    dims = len(data.mu[0])

    # Initialization:
    m = np.random.normal(size=(dims, K))
    s2 = np.ones(shape=(dims, K))
    phi = np.ones(shape=(n, K)) / K

    x = to_numpy(data.x)  # set data as (n,1) ndarray

    # implementation:
    is_conv = False
    conv_tol = 1e-5
    elbo = []

    while not is_conv:
        # update variational assignments:
        phi = np.exp(x @ m - 0.5 * (s2 + m**2).sum(axis=0))
        phi /= phi.sum(axis=1, keepdims=True)

        # update variational densities:
        den = 1/sigma**2 + phi.sum(axis=0, keepdims=True)
        m = (x.T @ phi) / den
        s2 = 1 / den

        elbo.append(calcElbo(x, sigma, m, s2, phi))
        if len(elbo) > 2 and np.abs(elbo[-2] - elbo[-1]) < conv_tol:
            is_conv = True

    if debug:
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(elbo)
        ax[0].set_xlabel("Iteration")
        ax[0].set_ylabel("ELBO(q)")
        ax[0].set_title("CAVI Convergence Plot")
        for k in range(K):
            k_data = data.loc[data.c == k]
            getCoord = lambda data, i: np.array([lst[i] for lst in data])
            ax[1].plot(getCoord(k_data.x, 0), getCoord(k_data.x, 1), 'o')
            mean = np.random.multivariate_normal(mean=m[:, k].T, cov=np.identity(dims) * s2[:, k])
            cov = np.identity(dims)
            samples_hat = np.random.multivariate_normal(mean=mean, cov=cov, size=int(1e5))
            confidence_ellipse(samples_hat[:, 0], samples_hat[:, 1], ax=ax[1],
                               facecolor="k", alpha=0.5, label="variational factors")

        ax[1].set_xlabel("x")
        ax[1].set_ylabel("Variational Approximating Factor")
        ax[1].legend()

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
    elbo_terms[0] = -1/(2*sigma**2) * (s2_hat + (m_hat**2).sum(axis=0)).sum()
    elbo_terms[1] = (phi_hat * (x @ m_hat - 0.5*(s2_hat + (m_hat**2)).sum(axis=0))).sum()
    elbo_terms[2] = -(phi_hat * np.log(phi_hat)).sum()
    elbo_terms[3] = -0.5*np.log(2 * s2_hat).sum()

    return elbo_terms.sum()

if __name__ == "__main__":
    sigma = 5
    K = 5
    data = getMogData(K=K, sigma=sigma, nSamples=int(1e3))

    # inspect ELBOs sensitivity to initialization:
    phi, m, s2, elbo = CAVI(data, 1, debug=True)
    fig, ax = plt.subplots()
    for i in range(10):
        phi, m, s2, elbo = CAVI(data, 1, debug=False)
        ax.plot(elbo, 'b')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Elbo(q)")
    ax.set_title("ELBO sensitivity to Different Initializations")
