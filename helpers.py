import random
import numpy as np


def f_1_5(X, theta, y):
    return np.sum((y - np.dot(X, theta)) ** 2)


def mh_1_5(X, theta, y, beta, num_iter):
    m = X.shape[0]
    counter = 0
    avg_f_difference = 0
    difference_counter = 0
    for i in range(num_iter):
        counter += 1
        if counter == 50000:
            break
        idx = np.random.randint(len(theta))
        theta_proposed = theta.copy()
        theta_proposed[idx] = 1 - theta_proposed[idx]
        f_theta_proposed = f_1_5(X, theta_proposed, y)
        f_theta = f_1_5(X, theta, y)
        avg_f_difference += f_theta_proposed - f_theta

        if f_theta_proposed < f_theta:
            theta = theta_proposed
            counter = 0
        else:
            acceptance_probability = np.exp(-beta * (f_theta_proposed - f_theta))
            if acceptance_probability >= np.random.uniform(0, 1):
                theta = theta_proposed
                counter = 0
            else:
                avg_f_difference += f_theta_proposed - f_theta
                difference_counter += 1
        if i%5000 == 0:
            print(f"m = {m}, beta = {beta}, Iteration #: {i}, f_theta = {f_theta}")
    return theta, avg_f_difference/difference_counter


def mh_2(X, theta, y, beta, num_iter):
    m = X.shape[0]
    counter = 0
    for i in range(num_iter):
        counter += 1
        if counter == 50000:
            break

        idx_i = np.random.randint(len(theta))
        idx_j = np.random.randint(len(theta))
        while theta[idx_i] == theta[idx_j]:
            idx_i = np.random.randint(len(theta))
            idx_j = np.random.randint(len(theta))
        theta_proposed = theta.copy()
        theta_proposed[idx_i] = theta[idx_j]
        theta_proposed[idx_j] = theta[idx_i]
        f_theta_proposed = f_1_5(X, theta_proposed, y)
        f_theta = f_1_5(X, theta, y)
        if f_theta_proposed < f_theta:
            theta = theta_proposed
            counter = 0
        else:
            acceptance_probability = np.exp(-beta * (f_theta_proposed - f_theta))
            if acceptance_probability >= np.random.uniform(0, 1):
                theta = theta_proposed
                counter = 0
        if i%5000 == 0:
            print(f"m = {m}, beta = {beta}, Iteration #: {i}, f_theta = {f_theta}")
    return theta


def simulated_annealing(X, theta, y, beta_2m_start, beta_2m_end, seq_length, num_iter):

    m = X.shape[0]
    beta_start = beta_2m_start / (2 * m)
    beta_end = beta_2m_end / (2 * m)

    result_sequence = geometric_sequence(beta_start, beta_end, seq_length)

    beta_index = 0
    beta = result_sequence[beta_index]
    counter = 0
    for i in range(num_iter):
        counter += 1
        if counter == 50000:
            break
        if i != 0 and i % (int(num_iter/seq_length)) == 0:
            beta_index += 1
            beta = result_sequence[beta_index]
        idx = np.random.randint(len(theta))
        theta_proposed = theta.copy()
        theta_proposed[idx] = 1 - theta_proposed[idx]
        f_theta_proposed = f_1_5(X, theta_proposed, y)
        f_theta = f_1_5(X, theta, y)

        if f_theta_proposed < f_theta:
            theta = theta_proposed
            counter = 0
        else:
            acceptance_probability = np.exp(-beta * (f_theta_proposed - f_theta))
            if acceptance_probability >= np.random.uniform(0, 1):
                theta = theta_proposed
                counter = 0
        if i%5000 == 0:
            print(f"m = {m}, beta = {beta}, Iteration #: {i}, f_theta = {f_theta}")
    return theta


def simulated_annealing_cont(X, theta, y, beta_2m_start, beta_2m_end, num_iter):

    m = X.shape[0]
    beta_start = beta_2m_start / (2 * m)
    beta_end = beta_2m_end / (2 * m)

    ratio = (beta_end / beta_start) ** (1 / (num_iter - 1))

    beta = beta_start
    counter = 0
    for i in range(num_iter):
        counter += 1
        if counter == 50000:
            break
        idx = np.random.randint(len(theta))
        theta_proposed = theta.copy()
        theta_proposed[idx] = 1 - theta_proposed[idx]
        f_theta_proposed = f_1_5(X, theta_proposed, y)
        f_theta = f_1_5(X, theta, y)

        if f_theta_proposed < f_theta:
            theta = theta_proposed
            counter = 0
        else:
            acceptance_probability = np.exp(-beta * (f_theta_proposed - f_theta))
            if acceptance_probability >= np.random.uniform(0, 1):
                theta = theta_proposed
                counter = 0
        if i%5000 == 0:
            print(f"m = {m}, beta = {beta}, Iteration #: {i}, f_theta = {f_theta}")
        beta = beta*ratio
    return theta


def mse_1_5(theta_true, theta_predicted):
    return (2/len(theta_true)) * np.sum((theta_predicted - theta_true)**2)


def accuracy_1_5(theta_true, theta_predicted):
    return np.sum(theta_predicted == theta_true) / len(theta_true)

def geometric_sequence(start, end, length):
    ratio = (end / start) ** (1 / (length - 1))
    sequence = [start * (ratio ** i) for i in range(length)]
    return sequence

