import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from pathlib import Path

figPath = Path("Figs")
import datetime


def genResDir():
    now = datetime.datetime.now()
    res_dir = "_".join([str(x) for x in [now.day, now.month, "", now.hour, now.minute, now.second, now.microsecond]])
    path = Path(figPath, res_dir)
    path.mkdir()
    return path


res_path = genResDir()


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


def plotData(data, ax, label):
    getCoord = lambda data_in, i: np.array([lst[i] for lst in data_in])
    ax.plot(getCoord(data.x, 0), getCoord(data.x, 1), 'o', label=label, alpha=0.5)


def plotPostPred(mu, sigma2, ax, label):
    """Plot the distribution of the Posterior Predictive"""
    dims = len(mu)
    mean = np.random.multivariate_normal(mean=mu.T, cov=np.identity(dims) * sigma2)
    cov = np.identity(dims)
    samples_hat = np.random.multivariate_normal(mean=mean, cov=cov, size=int(1e5))
    confidence_ellipse(samples_hat[:, 0], samples_hat[:, 1], ax=ax, facecolor="k", alpha=0.25, label=label)


def plotDist(mu, sigma2, ax, label, alpha=0.5):
    """Plot the distribution of the Gaussian detremined by mean mu_k and covariance sigma2"""
    dims = len(mu)
    samples_hat = np.random.multivariate_normal(mean=mu.T, cov=np.identity(dims) * sigma2, size=int(1e5))
    confidence_ellipse(samples_hat[:, 0], samples_hat[:, 1], ax=ax, facecolor="k", alpha=alpha, label=label)


def plotCaviProgress(data, m, s2, iter=None, path=res_path):
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

    if iter is None:
        suffix = "Initial State"
    else:
        suffix = f"Iteration #{iter}"

    fig.suptitle(f"CAVI Progress ({suffix})")
    fig.savefig(Path(path, f"CAVI_{suffix}.png"))
    plt.close(fig)


def to_numpy(seriesOfLists):
    listOfLists = [x for x in seriesOfLists]
    return np.array(listOfLists)

def genGif(path, dur=150):
    from PIL import Image
    flist = [x for x in path.iterdir() if x.match("*CAVI*")]
    flist.sort(key=lambda item: item.lstat().st_mtime)
    pil_list = [Image.open(img) for img in flist]
    for i in range(500 // dur):
        pil_list.insert(0, pil_list[0])
    pil_list[0].save(Path(path, "ilustration.gif"), save_all=True, append_images=pil_list[1:], loop=0, duration=dur)


if __name__=="__main__":
    genGif(Path("/home/omri/PycharmProjects/SignalProcessing2DataScience/FinalProject/Figs/sameData2/success2"))