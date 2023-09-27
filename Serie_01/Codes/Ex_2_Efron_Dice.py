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
    distribution_die1 = rv_discrete(values=die1)
    distribution_die2 = rv_discrete(values=die2)
    throws_die1 = distribution_die1.rvs(size=n_throws)
    throws_die2 = distribution_die2.rvs(size=n_throws)
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
    sample_mean = np.sum(throws) / n_throws
    sample_var = np.sum((throws - sample_mean) ** 2) / np.maximum(n_throws-1,1)
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
    true_mean = distribution.mean()
    true_var = distribution.var()
    sample_means = np.zeros(max_n_throws)
    sample_vars = np.zeros(max_n_throws)
    for i in range(1, max_n_throws):
        sample_means[i], sample_vars[i] = compute_sample_mean_var(throws[:i])
    x_val = np.arange(0, max_n_throws)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(x_val, sample_means, 'blue')
    ax[0].plot(x_val, true_mean * np.ones(max_n_throws), 'red')
    ax[0].set_title('True mean vs. sample mean')
    ax[1].plot(x_val, sample_vars, 'blue')
    ax[1].plot(x_val, true_var * np.ones(max_n_throws), 'red')
    ax[1].set_title('True variance vs. sample variance')
    plt.tight_layout()
    plt.show()
    return None


# --------------- Lab  -------------------------

# defining probabilities of dice
die_A = np.array([[0, 4], [1 / 3, 2 / 3]])
die_B = np.array([[3], [1]])
die_C = np.array([[2, 6], [2 / 3, 1 / 3]])
die_D = np.array([[1, 5], [1 / 2, 1 / 2]])


example_sample_A = rv_discrete(values=die_A).rvs(size=20)
print(example_sample_A)

print('Result of die C versus die D:', dice_game(die_C, die_D, 100))

example_sample_D = rv_discrete(values=die_D).rvs(size=50)
sample_mean, sample_var = compute_sample_mean_var(example_sample_D)
true_mean = rv_discrete(values=die_D).mean()
true_var = rv_discrete(values=die_D).var()
print('Sample', example_sample_D, 'of die D leads to sample mean=', sample_mean, 'and sample variance', sample_var)
print('True mean of die D=', true_mean, ', true variance of die D=', true_var)

plot_mean_var(die_D, 100)
