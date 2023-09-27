import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt


def monte_carlo_pi(num_of_pts):
    """
    Monte carlo simulation for the computation of pi

    Generate random points in the square [-1,1]x[-1,1]
    and count, how many of them lie in the unit circle

    Parameters
    ----------
    num_of_pts: int
        Number of points, which should be generated

    Returns
    -------
    pi_approximation: scalar
        Approximation of pi

    """
    # Code here
    # x_values =
    # y_values =
    distances = x_values ** 2 + y_values ** 2
    # Code here
    # counter=

    fig = plt.figure()
    ax = fig.gca()
    # Code here
    #
    plt.tight_layout()
    plt.show()

    return pi_approximation

# --------------- Lab  -------------------------

# print(monte_carlo_pi(1000))
