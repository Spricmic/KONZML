import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D


def multivariate_normal_computation(mu, sigma):
    """
    Computes points (x,y,z) corresponding to a
    multivariate normal distribution corresponding
    to a mean mu and covariance matrix sigma

    Parameters
    ----------
    mu: array
        Mean of the normal distribution
    sigma: array
        Covariance matrix of the normal distribution

    Returns
    -------
    x,y,z
        x: array of x coordinates
        y: array of y coordinates
        z: array of z coordinates

    """
    step = np.sqrt(np.amax(sigma))
    x, y = np.mgrid[mu[0] - 3 * step:mu[0] + 3 * step:.1, mu[1] - 3 * step:mu[1] + 3 * step:.1]
    pts = np.dstack((x, y))
    z = multivariate_normal.pdf(pts, mean=mu, cov=sigma)
    return x, y, z


def multivariate_normal_plot(mus, sigmas):
    """
    PLot the surface corresponding to the probability
    density function of a multivariate normal distribution

    Parameters
    ----------
    mus: array
        array of means
    sigmas: array
        array of covariance matrices

    Returns
    -------
    Plot: window

    """
    fig = plt.figure()
    n_distributions = mus.shape[0]
    for i in range(0, n_distributions):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        x, y, z = multivariate_normal_computation(mus[i], sigmas[i])
        surface = ax.plot_surface(x, y, z, cmap='coolwarm')
        fig.colorbar(surface)
        ax.set_title('With $ \mu=(0,0) $ and $ \sigma = ${}'.format(sigmas[i]))
    plt.show()
    return None


def multivariate_normal_contour(mus, sigmas):
    """
    PLot the contour plot corresponding to the probability
    density function of a multivariate normal distribution

    Parameters
    ----------
    mus: array
        array of means
    sigmas: array
        array of covariance matrices

    Returns
    -------
    Plot: window

    """
    fig = plt.figure()
    n_distributions = mus.shape[0]
    for i in range(0, n_distributions):
        ax = fig.add_subplot(2,3,i+1)
        x, y, z = multivariate_normal_computation(mus[i], sigmas[i])
        ax.contour(x, y, z, cmap='coolwarm')
        ax.set_title('With $ \mu=(0,0) $ and $ \sigma = ${}'.format(sigmas[i]))
    plt.show()
    return None


# --------------- Lab  -------------------------

# defining mus and sigmas
mu_vector = np.zeros((5, 2))
cov_1 = np.array([[1, 0], [0, 1]])
cov_2 = np.array([[2, 0], [0, 2]])
cov_3 = np.array([[3, 1], [1, 3]])
cov_4 = np.array([[3, 2.5], [2.5, 3]])
cov_5 = np.array([[0.8, 0.5], [0.5, 0.8]])
covariances = np.array([cov_1, cov_2, cov_3, cov_4, cov_5])

multivariate_normal_plot(mus=mu_vector, sigmas=covariances)
multivariate_normal_contour(mus=mu_vector, sigmas=covariances)