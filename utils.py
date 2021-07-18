import numpy as np
import scipy.stats as stats

def plotGauusian(mean, var, label, ax, *args):
    std = np.sqrt(var)
    x_span = np.linspace(mean - 3 * std, mean + 3 * std, 100)
    ax.plot(x_span, stats.norm.pdf(x_span, loc=mean, scale=std), label=label, *args)
