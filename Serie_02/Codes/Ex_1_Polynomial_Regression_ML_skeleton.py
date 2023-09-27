import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform


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
    sigma: scalar
        Standard deviation which determines noise

    Returns
    -------
    x_values, t_values
        v_values: list of x-values in [0,1]
        t_values: list of noisy t-values for sin(2*pi*x)

    """
    #Code here
    # x_values =

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

    # Code here

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
    #Code here

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
    degree: int
        Degree of the regression polynomial

    Returns
    -------
    Plot: Window

    """
    data_size = len(data_training[0])
    interval = np.linspace(0, 1, 100)
    sin_val = np.sin(2*np.pi*interval)

    #Code here
    #X =

    #Code here
    #fig = plt.figure()

    plt.show()
    return None


# --------------- Lab  -------------------------
if __name__ == '__main__':
    # n_data =
    # sigma =
    # degrees =

    # plot_noisy_data()

    # w_ML, var_ML = ml_polynomial_regression()
    # print('Learned variance:', var_ML, 'True variance:', sigma**2)

    # plot_ml_polynomial_regression()
