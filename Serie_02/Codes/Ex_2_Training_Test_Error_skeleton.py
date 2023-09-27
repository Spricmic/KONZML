import numpy as np
import matplotlib.pyplot as plt
from Ex_1_Polynomial_Regression_ML import generate_noisy_data, ml_polynomial_regression


def regression_polynomial_evaluation(coefficients, x_values):
    """
    Evaluates the regression polynomial for given coefficients and input values.

    The coefficients are assumed to be ordered increasing by degree,
    i.e. coefficients = [a_0, a_1, ..., a_n], where a_n is the coefficient corresponding to highest degree.

    Parameters
    ----------
    coefficients: array
        array of coefficients of polynomial
    x_values: array
        array of input values for the polynomial

    Returns
    -------
    poly_values: array of polynomial values, of same shape as input x_values

    """
    X = np.vander(x_values, len(coefficients), increasing=True)
    poly_values = np.matmul(X, coefficients)

    return poly_values


def plot_regression_polynomials(data_training, degrees):
    """
    Plot the regression polynomials of different degrees for given training data.

    Parameters
    ----------
    data_training: [x_values, t_values]
        x_values: array of input values of training data
        t_values: array of target values of training data
    degrees: array
        array of degree for regression polynomials

    Returns
    -------
    Plot: Window

    """
    interval = np.linspace(0, 1, 100)
    n_polynomials = len(degrees)

    fig, ax = plt.subplots(1, n_polynomials, figsize=(15, 8))
    for i in range(0, n_polynomials):
        #Code here
        #ax[i].plot()

    fig.suptitle('Fitting training data of size {} with polynomial of different degrees.'.format(len(data_training[0])))
    plt.show()

    return None


def rmse(true_values, estimated_values):
    """
    Compute the root mean square error between two array of same length.

    Parameters
    ----------
    true_values: array
        array of float of true values
    estimated_values: array
        array of float of estimated values

    Returns
    -------
    float: Root mean square error between input arrays

    """
    #Code here

    return np.sqrt(err)


def plot_errors(data_training, data_test, max_degree):
    """
    Plot the training and test error for regression polynomials of different degrees.

    Compute and plot training and test error for every polynomial of degree from 0 to max_degree.
    The error is the root mean square error and computed using the function rmse.

    Parameters
    ----------
    data_training: [x_values, t_values]
        x_values: array of input values of training data
        t_values: array of target values of training data
    data_test: [x_values, t_values]
        x_values: array of input values of test data
        t_values: array of target values of test data
    max_degree: int
        maximal degree of regression polynomial

    Returns
    -------
    Plot: Window

    """
    test_err = np.zeros(max_deg + 1)
    train_err = np.zeros(max_deg + 1)
    degrees = np.arange(0, max_degree + 1)
    for deg in range(0, max_deg + 1):
        #Code here
        #polynomial_coeff


    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca()
    #Code here
    #ax.plot()

    fig.suptitle('Training error vs. test error; training size={}, test size={}'.format(len(data_training[0]),
                                                                                        len(data_test[0])))
    plt.show()

    return None


# --------------- Lab  -------------------------
if __name__ == '__main__':
    # sigma =
    # train_size =
    # test_size =

    # train_set =
    # test_set =

    degrees = [1, 3, 9]
    # plot_regression_polynomials()

    # max_deg =
    # plot_errors()
