import random
import numpy as np
import scipy


def f_1(X, theta, y):
    return np.sum((y - np.dot(X, theta)) ** 2)


def f_3(X, theta, y):
    z = X @ theta / np.sqrt(2)  # Compute X_i^T * theta / sqrt(2)
    log_likelihood = np.log(1 + y * scipy.special.erf(z) + 1e-7)  # Compute log-likelihood for each observation
    neg_log_likelihood = -np.sum(log_likelihood)  # Sum up the negative log-likelihood values
    return neg_log_likelihood


def mh_1(X, theta, y, beta, num_iter):
    m = X.shape[0]
    counter = 0
    for i in range(num_iter):
        counter += 1
        if counter == 50000:
            break
        idx = np.random.randint(len(theta))
        theta_proposed = theta.copy()
        theta_proposed[idx] = 1 - theta_proposed[idx]
        f_theta_proposed = f_1(X, theta_proposed, y)
        f_theta = f_1(X, theta, y)

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


def mh_2(X, theta, y, beta, num_iter):
    m = X.shape[0]
    counter = 0
    for i in range(num_iter):
        counter += 1
        if counter == 50000:
            break
        zero_indices = np.where(theta == 0)[0]
        one_indices = np.where(theta == 1)[0]
        zero_index = np.random.choice(zero_indices)
        one_index = np.random.choice(one_indices)
        theta_proposed = theta.copy()
        theta_proposed[zero_index] = 1
        theta_proposed[one_index] = 0
        f_theta_proposed = f_1(X, theta_proposed, y)
        f_theta = f_1(X, theta, y)
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


def mh_3(X, theta, y, beta, num_iter):
    m = X.shape[0]
    counter = 0
    for i in range(num_iter):
        counter += 1
        if counter == 50000:
            break
        zero_indices = np.where(theta == 0)[0]
        one_indices = np.where(theta == 1)[0]
        zero_index = np.random.choice(zero_indices)
        one_index = np.random.choice(one_indices)
        theta_proposed = theta.copy()
        theta_proposed[zero_index] = 1
        theta_proposed[one_index] = 0
        f_theta_proposed = f_3(X, theta_proposed, y)
        f_theta = f_3(X, theta, y)
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

def simulated_annealing_mh_3(X, theta, y, beta_m_over_5_start, beta_m_over_5_end, seq_length, num_iter):

    m = X.shape[0]
    beta_start = beta_m_over_5_start * (5 / m)
    beta_end = beta_m_over_5_end * (5 / m)

    result_sequence = geometric_sequence(beta_start, beta_end, seq_length)
    seq_length = len(result_sequence)

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
        zero_indices = np.where(theta == 0)[0]
        one_indices = np.where(theta == 1)[0]
        zero_index = np.random.choice(zero_indices)
        one_index = np.random.choice(one_indices)
        theta_proposed = theta.copy()
        theta_proposed[zero_index] = 1
        theta_proposed[one_index] = 0
        f_theta_proposed = f_3(X, theta_proposed, y)
        f_theta = f_3(X, theta, y)

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


def simulated_annealing_mh_2(X, theta, y, beta_4m_start, beta_4m_end, seq_length, num_iter):

    m = X.shape[0]
    beta_start = beta_4m_start / (4 * m)
    beta_end = beta_4m_end / (4 * m)

    result_sequence = geometric_sequence(beta_start, beta_end, seq_length)
    seq_length = len(result_sequence)

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
        zero_indices = np.where(theta == 0)[0]
        one_indices = np.where(theta == 1)[0]
        zero_index = np.random.choice(zero_indices)
        one_index = np.random.choice(one_indices)
        theta_proposed = theta.copy()
        theta_proposed[zero_index] = 1
        theta_proposed[one_index] = 0
        f_theta_proposed = f_1(X, theta_proposed, y)
        f_theta = f_1(X, theta, y)

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


def simulated_annealing_mh_1(X, theta, y, beta_2m_start, beta_2m_end, seq_length, num_iter):

    m = X.shape[0]
    beta_start = beta_2m_start / (2 * m)
    beta_end = beta_2m_end / (2 * m)

    result_sequence = geometric_sequence(beta_start, beta_end, seq_length)
    seq_length = len(result_sequence)

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
        f_theta_proposed = f_1(X, theta_proposed, y)
        f_theta = f_1(X, theta, y)

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
        f_theta_proposed = f_1(X, theta_proposed, y)
        f_theta = f_1(X, theta, y)

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


def mh_3_avg_diff(X, theta, y, beta, num_iter):
    m = X.shape[0]
    counter = 0
    diff_counter = 0
    diff_sum = 0
    for i in range(num_iter):
        counter += 1
        if counter == 50000:
            break
        zero_indices = np.where(theta == 0)[0]
        one_indices = np.where(theta == 1)[0]
        zero_index = np.random.choice(zero_indices)
        one_index = np.random.choice(one_indices)
        theta_proposed = theta.copy()
        theta_proposed[zero_index] = 1
        theta_proposed[one_index] = 0
        f_theta_proposed = f_3(X, theta_proposed, y)
        f_theta = f_3(X, theta, y)
        if f_theta_proposed < f_theta:
            theta = theta_proposed
            counter = 0
        else:
            diff_counter += 1
            diff_sum += f_theta_proposed - f_theta
            acceptance_probability = np.exp(-beta * (f_theta_proposed - f_theta))
            if acceptance_probability >= np.random.uniform(0, 1):
                theta = theta_proposed
                counter = 0
        if i%5000 == 0:
            print(f"m = {m}, beta = {beta}, Iteration #: {i}, f_theta = {f_theta}")
    return diff_sum/diff_counter


def mse_1(theta_true, theta_predicted):
    return (2/len(theta_true)) * np.sum((theta_predicted - theta_true)**2)


def mse_sparse(theta_true, theta_predicted, s):
    return (1/(2*s)) * np.sum((theta_predicted - theta_true)**2)


def geometric_sequence(start, end, length):
    ratio = (end / start) ** (1 / (length - 1))
    sequence = [start * (ratio ** i) for i in range(length)]
    return sequence

def mh_competition(X, theta, y, beta, num_iter):
    m = X.shape[0]
    counter = 0
    for i in range(num_iter):
        counter += 1
        if counter == 50000:
            break

        zero_indices = np.where(theta == 0)[0]
        nonzero_indices = np.where((theta == 1) | (theta == 2))[0]

        zero_index = np.random.choice(zero_indices)
        nonzero_index = np.random.choice(nonzero_indices)
        theta_proposed = theta.copy()
        theta_proposed[zero_index] = np.random.choice([1, 2])
        theta_proposed[nonzero_index] = 0


        f_theta_proposed = f_1(X, theta_proposed, y)
        f_theta = f_1(X, theta, y)
        if f_theta_proposed < f_theta:
            theta = theta_proposed
            counter = 0
        else:
            acceptance_probability = np.exp(-beta * (f_theta_proposed - f_theta))
            if acceptance_probability >= np.random.uniform(0, 1):
                theta = theta_proposed
                counter = 0
        if i % 5000 == 0:
            print(f"m = {m}, beta = {beta}, Iteration #: {i}, f_theta = {f_theta}")
    return theta

def mh_competition_SA(X, theta, y, beta_4m_start, beta_4m_end, seq_length, num_iter):
    m = X.shape[0]
    counter = 0

    beta_start = beta_4m_start / (4 * m)
    beta_end = beta_4m_end / (4 * m)

    result_sequence = geometric_sequence(beta_start, beta_end, seq_length)
    seq_length = len(result_sequence)

    beta_index = 0
    beta = result_sequence[beta_index]

    for i in range(num_iter):
        counter += 1
        if counter == 50000:
            break
        if i != 0 and i % (int(num_iter/seq_length)) == 0:
            beta_index += 1
            beta = result_sequence[beta_index]

        one_indices = np.where(theta == 1)[0]
        two_indices = np.where(theta == 2)[0]

        case_array = []
        if len(one_indices) > 0:
            case_array.append(1)
        if len(two_indices) > 0:
            case_array.append(2)
        if len(one_indices) > 0 and len(two_indices) > 0:
            case_array.append(3)

        case = np.random.choice(case_array)
        if case == 1:
            zero_indices = np.where(theta == 0)[0]
            zero_index = np.random.choice(zero_indices)
            one_index = np.random.choice(one_indices)

            theta_proposed = theta.copy()
            theta_proposed[zero_index] = 1
            theta_proposed[one_index] = 0

        elif case == 2:
            zero_indices = np.where(theta == 0)[0]
            zero_index = np.random.choice(zero_indices)
            two_index = np.random.choice(two_indices)

            theta_proposed = theta.copy()
            theta_proposed[zero_index] = 2
            theta_proposed[two_index] = 0

        else:
            one_index = np.random.choice(one_indices)
            two_index = np.random.choice(two_indices)

            theta_proposed = theta.copy()
            theta_proposed[one_index] = 2
            theta_proposed[two_index] = 1

        f_theta_proposed = f_1(X, theta_proposed, y)
        f_theta = f_1(X, theta, y)

        if f_theta_proposed < f_theta:
            theta = theta_proposed
            counter = 0
        else:
            acceptance_probability = np.exp(-beta * (f_theta_proposed - f_theta))
            if acceptance_probability >= np.random.uniform(0, 1):
                theta = theta_proposed
                counter = 0
        if i % 5000 == 0:
            print(f"m = {m}, beta = {beta}, Iteration #: {i}, f_theta = {f_theta}")
    return theta

