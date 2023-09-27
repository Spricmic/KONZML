# import packages
import numpy as np
from scipy.stats import rv_discrete
import matplotlib.pyplot as plt


# simulating game between two dice
def dice_game(die1, die2, n_throws):
    """
    Simulates multiple games between two dice and counts,
    how often each of them wins

    Parameters
    ----------
    die1 : array of [outcomes, probabilities]
        Contains information about die1
        outcomes: possible outcomes of die1
        probabilities: corresponding probabilities
    die2 : array of [outcomes, probabilities]
        Contains information about die2
        outcomes: array of int
            possible outcomes of die2
        probabilities: array of scalars
            corresponding probabilities
    n_throws : int
        Number of throws

    Returns
    -------
    counter: array of [win1, win2]
        Contains information about game
        win1: int
            number of games won by die1
        win2: int
            number of games won by die2

    """
    counter = np.zeros(2, dtype=int)
    # Code here
    # distribution_die1 =
    #
    # throws_die1 =
    #
    for i in range(0, n_throws):
        if throws_die1[i] > throws_die2[i]:
            counter[0] += 1
        else:
            counter[1] += 1
    return counter


def compute_sample_mean_var(throws):
    """
    Computes the sample mean and variance of
    a sequence of die throws

    Parameters
    ----------
    throws: array
        A sequence of the outcomes of a thrown die

    Returns
    -------
    sample_mean: scalar
        The sample mean of the thrown sequence
    sample_var: scalar
        The sample variance of the thrown sequence

    """
    n_throws = len(throws)
    # Code here
    #
    return sample_mean, sample_var


def plot_mean_var(die, max_n_throws):
    """
    Simulates a number of die throws and computes
    successively the mean and variance for each new throw;
    plots the results and the true mean and variance

    Parameters
    ----------
    die : array of [outcomes, probabilities]
        Contains information about die
        outcomes: array of int
            possible outcomes of die
        probabilities: array
            corresponding probabilities
    max_n_throws : int
        Maximal number of throws

    Returns
    -------
    Plot: window

    """
    distribution = rv_discrete(values=die)
    throws = distribution.rvs(size=max_n_throws)
    # Code here
    # true_mean =
    # true_var =
    sample_means = np.zeros(max_n_throws)
    sample_vars = np.zeros(max_n_throws)
    for i in range(1, max_n_throws):
        sample_means[i], sample_vars[i] = compute_sample_mean_var(throws[:i])
    x_val = np.arange(0, max_n_throws)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(x_val, sample_means, 'blue')
    ax[0].plot(x_val, true_mean * np.ones(max_n_throws), 'red')
    ax[0].set_title('True mean vs. sample mean')
    # Code here
    # ax[1].plot()
    #

    plt.tight_layout()
    plt.show()
    return None


# --------------- Lab  -------------------------

# defining probabilities of dice
die_A = np.array([[0, 4], [1 / 3, 2 / 3]])
# Code here
# die_B =
#
