# import packages
import numpy as np
from scipy.stats import bernoulli, binom, norm, beta
import matplotlib.pyplot as plt


def random_sample_bernoulli(p, n_throws):
    """
    Generate a random sample of size n_throws, bernoulli distributed
    with parameter p; plot the sample

    Parameters
    ----------
    p : float
        Parameter of bernoulli distribution
    n_throws : int
        Number of coin throws, i.e. size of sample

    Returns
    -------
    Plot: Window

    """
    sample = bernoulli.rvs(p, size=n_throws)
    print('Random sample from bernoulli with p =', p, ':', sample)

    x = np.arange(0, n_throws)

    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(x, sample)
    ax.set_xlabel('Throw number')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Tails', 'Heads'])
    ax.set_ylabel('Head/Tails')
    ax.set_title('{} throws of a biased coin with p={} probability of throwing heads'.format(n_throws, p))
    plt.show()
    return None


def random_sample_binom(n, p, sample_size):
    """
    Generate a random sample of size sample_size, binomially
    distributed with parameters n and p;
    plot the sample and the probability mass function

    Parameters
    ----------
    n : int
        Defines support {0, 1, ..., n} of binomial distribution
    p : float
        Parameter of binomial distribution
    sample_size : int
        Size of sample

    Returns
    -------
    Plot: Window

    """
    #Code here
    #sample =
    print('Random sample of size =', sample_size, 'from binom with n =', n, 'and p =', p, ':', sample)

    x_values = np.arange(0, n + 1)
    sample_counted = np.array([np.sum(sample == i) for i in range(0, n + 1)])

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    #Code here
    #ax[0].bar()

    plt.tight_layout()
    plt.show()
    return None


def random_sample_normal(mu, sigma, sample_size, n_bins):
    """
    Generate a random sample of size sample_size, normally
    distributed with mean mu and standard deviation sigma;
    plot the sample and the probability density function

    Parameters
    ----------
    mu : scalar
        Defines mean of the normal distribution
    sigma : scalar
        Standard deviation of the normal distribution
    sample_size : int
        Size of sample
    n_bins: int
        Number of bins, in which the sample gets divided

    Returns
    -------
    Plot: Window

    """
    #Code here
    #sample =
    x_vals = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)

    fig, ax = plt.subplots(2, 1, sharex=True)
    #Code here
    #ax[0].hist()

    fig.suptitle('Normal distribution with mean $ \mu=${} and standard deviation $ \sigma= ${}'.format(mu, sigma))
    plt.tight_layout()
    plt.show()
    return None


def plot_beta_pdf(parameters_a_b):
    """
    Generates the plots of beta distributions for different
    parameter-pairs (a,b)

    Parameters
    ----------
    parameters_a_b : array
        array contains pairs [a, b]

    Returns
    -------
    Plot: Window

    """
    n_pairs = parameters_a_b.shape[0]
    x_values = np.linspace(0, 1, 100)
    y_values = np.zeros((n_pairs, 100))
    fig = plt.figure()
    ax = fig.gca()

    for i in range(0, n_pairs):
        #Code here
        y_values[i, :] = 0

    ax.legend()
    plt.show()
    return None


def plot_beta_mean_median(a, b):
    """
    Plot the probability density function, mean and
    median of a beta(a,b) distribution

    Parameters
    ----------
    a : scalar
        Parameter for beta distribution
    b : scalar
        Parameter for beta distribution

    Returns
    -------
    Plot: Window

    """
    x_values = np.linspace(0, 1, 100)
    #Code here
    #y_values =

    fig = plt.figure()
    ax = fig.gca()
    #Code here
    #ax.plot()

    ax.legend()
    plt.show()
    return None


# --------------- Lab  -------------------------

random_sample_bernoulli(0.3, 30)
# random_sample_binom()
# random_sample_normal()
# plot_beta_pdf()
# plot_beta_mean_median()
