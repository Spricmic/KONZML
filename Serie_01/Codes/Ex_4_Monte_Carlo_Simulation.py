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
    x_values = uniform.rvs(loc=-1, scale=2, size=num_of_pts)
    y_values = uniform.rvs(loc=-1, scale=2, size=num_of_pts)
    distances = x_values ** 2 + y_values ** 2
    distances_binary = np.array(distances <= 1)
    counter = sum(distances_binary)
    pi_approximation = 4 * counter / num_of_pts

    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect('equal')
    ax.scatter(x_values, y_values, c=distances_binary)
    fig.suptitle('Monte Carlo simulation for $ \pi $ with {} points'.format(num_of_pts))
    plt.tight_layout()
    plt.show()

    return pi_approximation


# --------------- Lab  -------------------------

print(monte_carlo_pi(10000))
