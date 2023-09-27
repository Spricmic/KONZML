import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform

from collections.abc import Iterable


def generate_noisy_data(sample_size, sigma):
    """
    Generate noised data for sin(2*pi*x) in [0,1]

    Generate uniformly distributed x_values in the interval [0,1].
    Calculate function values using function.
    Add independent normally distributed zero mean noise with standard deviation sigma.

    Parameters
    ----------
    sample_size : int
        Size of sample data
    sigma: float
        Standard deviation which determines noise

    Returns
    -------
    x_values, t_values
        v_values: list of x-values in [0,1]
        t_values: list of noisy t-values for sin(2*pi*x)

    """
    x_values = uniform.rvs(size=sample_size)
    noise = norm.rvs(loc=0, scale=sigma, size=sample_size)
    t_values = np.sin(2 * np.pi * x_values) + noise
    return x_values, t_values


def plot_noisy_data(data):
    """
    Plot noisy data and sin(2*pi*x) in [0, 1]

    Parameters
    ----------
    data: [x_values, t_values]
        x_values: array of input values of data
        t_values: array of target values of data

    Returns
    -------
    Plot: Window

    """
    interval = np.linspace(0, 1, 100)
    sin_values = np.sin(2*np.pi*interval)
    x_values = data[0]
    t_values = data[1]

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(interval, sin_values, color='blue', label='$\sin(2 \pi x)$')
    ax.scatter(x_values, t_values, color='orange', marker='o', label='Noisy data', s = 150)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    ax.legend()
    plt.show()
    return None


def ml_polynomial_regression(data_training, degree):
    """
    Compute parameters for polynomial regression using maximum likelihood.

    Compute the coefficients w_k of the regression polynomial of given degree and the variance
    of the Gaussian distribution, which is used for the probabilistic model.

    Parameters
    ----------
    data_training: [x_values, t_values]
        x_values: array of input values of training data
        t_values: array of target values of training data
    degree: int
        Degree of the regression polynomial

    Returns
    -------
    w_ML, var_ML
        w_ML: array of coefficients of regression polynomial
        var_ML: Variance used in model

    """
    x_training = data_training[0]
    t_training = data_training[1]
    N = x_training.shape[0]
    X = np.vander(x_training, degree+1, increasing=True)
    # w_ML = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), t_training) dont solve least squares by formula
    w_ML = np.linalg.lstsq(X, t_training, rcond=None)[0]
    var_ML = np.linalg.norm(t_training - np.matmul(X, w_ML))**2 / N
    return w_ML, var_ML


def plot_ml_polynomial_regression(data_training, degree):
    """
    PLot the regression polynomial of given degree for a given set of data points.

    The coefficients of the polynomial are learned using maximum likelihood.

    Parameters
    ----------
    data_training: [x_values, t_values]
        x_values: array of input values of training data
        t_values: array of target values of training data
    degree: list of ints
        Degrees of the regression polynomials

    Returns
    -------
    Plot: Window

    """
    data_size = len(data_training[0])
    interval = np.linspace(0, 1, 1000)
    sin_val = np.sin(2*np.pi*interval)

    if not isinstance(degree, Iterable):
        degree = [degree]
    nb_cols = int(np.ceil(np.sqrt(len(degree))))
    nb_rows = int(np.ceil(len(degree)/nb_cols))
    fig = plt.figure()
    fig.suptitle('Polynomial ML-Regression', fontsize=16)

    for sp, deg in enumerate(degree):
        X = np.vander(interval, deg+1, increasing=True)
        w_ML, var_ML = ml_polynomial_regression(data_training, deg)
        print('Degree ',deg,'; Learned coefficients:', w_ML, '; Learned variance:', var_ML, '; True variance:', sigma**2)
        regression_poly_values = np.matmul(X, w_ML)

        ax = fig.add_subplot(nb_rows, nb_cols, sp+1)
        ax.plot(interval, sin_val, color='blue', label='$\sin(2 \pi x)$')
        ax.scatter(data_training[0], data_training[1], color='orange', marker='o', label='Data of size {}'.format(data_size),s = 150)
        ax.plot(interval, regression_poly_values, color='red', label='Polynomial fit of degree M= {}'.format(deg))
        ax.set_ylim((-1.3, 1.3))
        ax.legend()
    plt.show()
    return None

# --------------- Lab  -------------------------
if __name__ == '__main__':

    n = 10
    sigma = 0.3
    data = generate_noisy_data(n, sigma)
    plot_noisy_data(data)
    degrees = [0, 1, 3, 9]
    plot_ml_polynomial_regression(data, degrees)
