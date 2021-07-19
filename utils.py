import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from pathlib import Path
import datetime
import pandas as pd

figPath = Path("Figs")
if not figPath.is_dir():
    figPath.mkdir()


def genResDir(figPath):
    now = datetime.datetime.now()
    res_dir = "_".join([str(x) for x in [now.day, now.month, "", now.hour, now.minute, now.second, now.microsecond]])
    path = Path(figPath, res_dir)
    path.mkdir()
    return path


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def getMogData(K: int = 5, sigma: float = 1, nDims: int = 2, nSamples: int = 100, debug: bool = True):
    """Generates a MOG model and draws nSamples from it"""
    mu_k = np.random.multivariate_normal(mean=np.zeros(shape=nDims), cov=sigma ** 2 * np.identity(nDims), size=K)
    c = np.random.choice(range(K), size=nSamples, p=np.ones(shape=K) / K)
    mu_c = mu_k[c]
    cov_k = np.zeros((K, nDims, nDims))
    for k in range(K):
        #     randSquaredMat = np.random.normal(size=(nDims, nDims))
        #     cov_k[k] = randSquaredMat.T @ randSquaredMat
        #     # cov_k[k] = make_spd_matrix(n_dim=nDims)
        cov_k[k] = np.identity(nDims)

    x_mu_c = lambda i: np.random.multivariate_normal(mean=mu_c[i].squeeze(axis=0), cov=cov_k[c[i]].squeeze(axis=0))
    x = np.apply_along_axis(x_mu_c, axis=1, arr=np.expand_dims(np.arange(nSamples),
                                                               axis=1))  # np.random.multivariate_normal(mu_c, cov=np.identity(nDims))
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
            plotData(k_data, ax[1], f"k={k}")

            mean = np.array(k_data.mu.iloc[0])
            cov = np.array(k_data.loc[:, "cov"].iloc[0])
            getCoord = lambda data, i: np.array([lst[i] for lst in data])
            confidence_ellipse(getCoord(k_data.x, 0), getCoord(k_data.x, 1), ax=ax[1],
                               facecolor=ax[1].lines[-1].get_color(), alpha=0.5)

        # ax[1].axes.set_aspect('equal')
        ax[1].legend()
        ax[1].set_title("Sampled Data")
        ax[1].set_xlabel("$x$")
        ax[1].set_ylabel("$p(x|c, \mu)$")
        # ax[1].set_ylim((0, 0.5))
        ax[1].grid()

    return x_df


def plotData(data, ax, label):
    getCoord = lambda data_in, i: np.array([lst[i] for lst in data_in])
    ax.plot(getCoord(data.x, 0), getCoord(data.x, 1), 'o', label=label, alpha=0.5)


def plotDist(mu, sigma2, ax, label, alpha=0.5):
    """Plot the distribution of the Gaussian detremined by mean mu_k and covariance sigma2"""
    dims = len(mu)
    samples_hat = np.random.multivariate_normal(mean=mu.T, cov=np.identity(dims) * sigma2, size=int(1e5))
    confidence_ellipse(samples_hat[:, 0], samples_hat[:, 1], ax=ax, facecolor="k", alpha=alpha, label=label)


def plotCaviState(data, m, s2, iter=None):
    K = len(data.c.unique())
    fig, ax = plt.subplots(2, 1)
    for k in range(K):
        k_data = data.loc[data.c == k]

        # Plot Gaussian Variational Factor
        plotData(k_data, ax[0], f"k={k}")
        dist_label = "$\mathcal{N}(\mu_k;m_k, s^2_k)$" if k == 0 else None
        plotDist(m[:, k], s2[:, k], ax[0], dist_label, alpha=0.7)

        # Plot Posterior Predictive:
        plotData(k_data, ax[1], f"k={k}")
        dist_label = "$\mathcal{N}(\mu_k, 1)$" if k == 0 else None
        plotDist(m[:, k], 1, ax[1], dist_label, alpha=0.25)

    data_np = to_numpy(data.x)
    for axis in ax:
        axis.set_xlabel("$x_1$")
        axis.set_ylabel("$x_2$")
        axis.set_xlim([data_np[:, 0].min() * 1.1, data_np[:, 0].max() * 1.1])
        axis.set_ylim([data_np[:, 1].min() * 1.1, data_np[:, 1].max() * 1.1])
        axis.legend()

    ax[0].set_title("Variational Factors $q(\mu_k)$")
    ax[1].set_title(r"Posterior Predictive Approximation $p(x_{new}|x) \approx \frac{1}{K}\sum_1^kp(x_{new};m_k)$")

    if iter == 0 :
        suffix = "Initial State"
    elif iter == -1:
        suffix = "Final State"
    else:
        suffix = f"Iteration #{iter}"

    fig.suptitle(f"CAVI Progress ({suffix})")

    return fig, ax

def plotCaviIter(data, debug_data, iter=None):
    """Plot the CAVI state at a specific iteration"""
    if iter is None:
        iter = 0
    m = debug_data.m.iloc[iter]
    s2 = debug_data.s2.iloc[iter]
    return plotCaviState(data, m, s2, iter)



def to_numpy(seriesOfLists):
    listOfLists = [x for x in seriesOfLists]
    return np.array(listOfLists)


def genGifFromPlots(path, dur=150):
    from PIL import Image
    flist = [x for x in path.iterdir() if x.match("*Cavi*")]
    flist.sort(key=lambda item: item.lstat().st_mtime)
    pil_list = [Image.open(img) for img in flist]
    for i in range(500 // dur):
        pil_list.insert(0, pil_list[0])
    pil_list[0].save(Path(path, "ilustration.gif"), save_all=True, append_images=pil_list[1:], loop=0, duration=dur)



def makeGif(samples, debug_data):
    figPath = Path("Figs")
    if not figPath.is_dir():
        figPath.mkdir()
    res_dir = genResDir(figPath)
    for iter in tqdm(range(len(debug_data)), desc="Processing CAVI Iterations"):
        fig, _ = plotCaviIter(samples, debug_data, iter)
        fig.savefig(Path(res_dir, f"Cavi_iter{iter}.png"))
        plt.close(fig)
    print("Generating GIF...")
    genGifFromPlots(res_dir)
    print("GIF is ready!")
    return res_dir

if __name__ == "__main__":
    genGifFromPlots(Path("/home/omri/PycharmProjects/SignalProcessing2DataScience/FinalProject/Figs/sameData2/success2"))
