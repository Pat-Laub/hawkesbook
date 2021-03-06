# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as rnd
from scipy.optimize import fsolve, minimize

from tqdm import tqdm
from numba import njit, prange


@njit()
def numba_seed(seed):
    rnd.seed(seed)


# Intensities and compensators


def hawkes_intensity(t, 鈩媉t, 饾泬):
    位, 渭, _ = 饾泬
    位耍 = 位
    for t_i in 鈩媉t:
        位耍 += 渭(t - t_i)
    return 位耍


def hawkes_compensator(t, 鈩媉t, 饾泬):
    if t <= 0: return 0
    位, _, M = 饾泬

    螞 = 位 * t
    for t_i in 鈩媉t:
        螞 += M(t - t_i)
    return 螞


def exp_hawkes_intensity(t, 鈩媉t, 饾泬):
    位, 伪, 尾 = 饾泬
    位耍 = 位
    for t_i in 鈩媉t:
        位耍 += 伪 * np.exp(-尾 * (t - t_i))
    return 位耍


def exp_hawkes_compensator(t, 鈩媉t, 饾泬):
    if t <= 0: return 0
    位, 伪, 尾 = 饾泬
    螞 = 位 * t
    for t_i in 鈩媉t:
        螞 += (伪/尾) * (1 - np.exp(-尾*(t - t_i)))
    return 螞


@njit(nogil=True)
def exp_hawkes_compensators(鈩媉t, 饾泬):
    位, 伪, 尾 = 饾泬

    螞 = 0
    位耍_prev = 位
    t_prev = 0

    螞s = np.empty(len(鈩媉t), dtype=np.float64)
    for i, t_i in enumerate(鈩媉t):
        螞 += 位 * (t_i - t_prev) + (
                (位耍_prev - 位)/尾 *
                (1 - np.exp(-尾*(t_i - t_prev))))
        螞s[i] = 螞

        位耍_prev = 位 + (位耍_prev - 位) * (
                np.exp(-尾 * (t_i - t_prev))) + 伪
        t_prev = t_i
    return 螞s


@njit(nogil=True)
def power_hawkes_intensity(t, 鈩媉t, 饾泬):
    位, k, c, p = 饾泬
    位耍 = 位
    for t_i in 鈩媉t:
        位耍 += k / (c + (t-t_i))**p
    return 位耍


@njit(nogil=True)
def power_hawkes_compensator(t, 鈩媉t, 饾泬):
    位, k, c, p = 饾泬
    螞 = 位 * t
    for t_i in 鈩媉t:
        螞 += ((k * (c * (c + (t-t_i)))**-p *
              (-c**p * (c + (t-t_i)) + c * (c + (t-t_i))**p)) /
              (p - 1))
    return 螞


@njit(nogil=True, parallel=True)
def power_hawkes_compensators(鈩媉t, 饾泬):
    螞s = np.empty(len(鈩媉t), dtype=np.float64)
    for i in prange(len(鈩媉t)):
        t_i = 鈩媉t[i]
        鈩媉i = 鈩媉t[:i]
        螞s[i] = power_hawkes_compensator(t_i, 鈩媉i, 饾泬)
    return 螞s


# Likelihood

def log_likelihood(鈩媉T, T, 饾泬, 位耍, 螞):
    鈩? = 0.0
    for i, t_i in enumerate(鈩媉T):
        鈩媉i = 鈩媉T[:i]
        位耍_i = 位耍(t_i, 鈩媉i, 饾泬)
        鈩? += np.log(位耍_i)
    鈩? -= 螞(T, 鈩媉T, 饾泬)
    return 鈩?


@njit(nogil=True, parallel=True)
def power_log_likelihood(鈩媉T, T, 饾泬):
    鈩? = 0.0
    for i in prange(len(鈩媉T)):
        t_i = 鈩媉T[i]
        鈩媉i = 鈩媉T[:i]
        位耍_i = power_hawkes_intensity(t_i, 鈩媉i, 饾泬)
        鈩? += np.log(位耍_i)
    鈩? -= power_hawkes_compensator(T, 鈩媉T, 饾泬)
    return 鈩?


@njit()
def exp_log_likelihood(鈩媉T, T, 饾泬):
    位, 伪, 尾 = 饾泬
    饾惌 = 鈩媉T
    N_T = len(饾惌)

    A = np.empty(N_T, dtype=np.float64)
    A[0] = 0
    for i in range(1, N_T):
        A[i] = np.exp(-尾*(饾惌[i] - 饾惌[i-1])) * (1 + A[i-1])

    鈩? = -位*T
    for i, t_i in enumerate(鈩媉T):
        鈩? += np.log(位 + 伪 * A[i]) - \
                (伪/尾) * (1 - np.exp(-尾*(T-t_i)))
    return 鈩?


def exp_mle(饾惌, T, 饾泬_start=np.array([1.0, 2.0, 3.0])):
    eps = 1e-5
    饾泬_bounds = ((eps, None), (eps, None), (eps, None))
    loss = lambda 饾泬: -exp_log_likelihood(饾惌, T, 饾泬)
    饾泬_mle = minimize(loss, 饾泬_start, bounds=饾泬_bounds).x
    return np.array(饾泬_mle)


def power_mle(饾惌, T, 饾泬_start=np.array([1.0, 1.0, 2.0, 3.0])):
    eps = 1e-5
    饾泬_bounds = ((eps, None), (eps, None), (eps, None),
        (1+eps, 100))
    loss = lambda 饾泬: -power_log_likelihood(饾惌, T, 饾泬)
    饾泬_mle = minimize(loss, 饾泬_start, bounds=饾泬_bounds).x
    return np.array(饾泬_mle)


# Simulation


def simulate_inverse_compensator(饾泬, 螞, N):
    鈩? = np.empty(N, dtype=np.float64)

    t耍_1 = -np.log(rnd.rand())
    exp_1 = lambda t_1: 螞(t_1, 鈩媅:0], 饾泬) - t耍_1

    t_1_guess = 1.0
    t_1 = fsolve(exp_1, t_1_guess)[0]

    鈩媅0] = t_1
    t_prev = t_1
    for i in range(1, N):
        螖t耍_i = -np.log(rnd.rand())

        螞_i = 螞(t_prev, 鈩?, 饾泬)
        exp_i = lambda t_next: 螞(t_next, 鈩媅:i], 饾泬) - 螞_i - 螖t耍_i

        t_next_guess = t_prev + 1.0
        t_next = fsolve(exp_i, t_next_guess)[0]

        鈩媅i] = t_next
        t_prev = t_next
    return 鈩?

@njit(nogil=True)
def exp_simulate_by_composition(饾泬, N):
    位, 伪, 尾 = 饾泬
    位耍_k = 位
    t_k = 0

    鈩? = np.empty(N, dtype=np.float64)
    for k in range(N):
        U_1 = rnd.rand()
        U_2 = rnd.rand()

        # Technically the following works, but without @njit
        # it will print out "RuntimeWarning: invalid value encountered in log".
        # This is because 1 + 尾/(位耍_k + 伪 - 位)*np.log(U_2) can be negative
        # so T_2 can be np.NaN. The Dassios & Zhao (2013) algorithm checks if this
        # expression is negative and handles it separately, though the lines
        # below have the same behaviour as t_k = min(T_1, np.NaN) will be T_1. 
        T_1 = t_k - np.log(U_1) / 位
        T_2 = t_k - np.log(1 + 尾/(位耍_k + 伪 - 位)*np.log(U_2))/尾

        t_prev = t_k
        t_k = min(T_1, T_2)
        鈩媅k] = t_k

        if k > 0:
            位耍_k = 位 + (位耍_k + 伪 - 位) * (
                np.exp(-尾 * (t_k - t_prev)))
        else:
            位耍_k = 位
          
    return 鈩?


@njit(nogil=True)
def exp_simulate_by_thinning(饾泬, T):
    位, 伪, 尾 = 饾泬

    位耍 = 位
    times = []

    t = 0

    while True:
        M = 位耍
        螖t = rnd.exponential() / M
        t += 螖t
        if t > T:
            break

        位耍 = 位 + (位耍 - 位) * np.exp(-尾 * 螖t)

        u = M * rnd.rand()
        if u > 位耍:
            continue  # This potential arrival is 'thinned' out

        times.append(t)
        位耍 += 伪

    return np.array(times)


@njit(nogil=True)
def power_simulate_by_thinning(饾泬, T):
    位, k, c, p = 饾泬

    位耍 = 位
    times = []

    t = 0

    while True:
        M = 位耍
        螖t = rnd.exponential() / M
        t += 螖t
        if t > T:
            break

        位耍 = power_hawkes_intensity(t, np.array(times), 饾泬)

        u = M * rnd.rand()
        if u > 位耍:
            continue  # This potential arrival is 'thinned' out

        times.append(t)
        位耍 += k / (c ** p)

    return np.array(times)


# Moment matching


def empirical_moments(饾惌, T, 蟿, lag):
    bins = np.arange(0, T, 蟿)
    N = len(bins) - 1
    count = np.zeros(N)

    for i in range(N):
        count[i] = np.sum((bins[i] <= 饾惌) & (饾惌 < bins[i+1]))

    empMean = np.mean(count)
    empVar = np.std(count)**2
    empAutoCov = np.mean((count[:-lag] - empMean) \
                    * (count[lag:] - empMean))

    return np.array([empMean, empVar, empAutoCov]).reshape(3,1)



def exp_moments(饾泬, 蟿, lag):
    """
    Consider an exponential Hawkes process with parameter 饾泬.
    Look at intervals of length 蟿, i.e. N(t+蟿) - N(t).
    Calculate the limiting (t->鈭?) mean and variance.
    Also, get the limiting autocovariance:
        E[ (N(t + 蟿) - N(t)) (N(t + lag*蟿 + 蟿) - N(t + lag*蟿)) ].
    """
    位, 伪, 尾 = 饾泬
    魏 = 尾 - 伪
    未 = lag*蟿

    mean = (位*尾/魏)*蟿
    var = (位*尾/魏)*(蟿*(尾/魏) + (1 - 尾/魏)*((1 - np.exp(-魏*蟿))/魏))
    autoCov = (位*尾*伪*(2*尾-伪)*(np.exp(-魏*蟿) - 1)**2/(2*魏**4)) \
                *np.exp(-魏*未)

    return np.array([mean, var, autoCov]).reshape(3,1)


def exp_gmm_loss(饾泬, 蟿, lag, empMoments, W):
    moments = exp_moments(饾泬, 蟿, lag)
    饾悹 = empMoments - moments
    return (饾悹.T).dot(W).dot(饾悹)[0,0]

def exp_gmm(饾惌, T, 蟿=5, lag=5, iters=2, 饾泬_start=np.array([1.0, 2.0, 3.0])):
    empMoments = empirical_moments(饾惌, T, 蟿, lag)

    W = np.eye(3)
    bounds = ((0, None), (0, None), (0, None))

    饾泬 = minimize(exp_gmm_loss, x0=饾泬_start,
            args=(蟿, lag, empMoments, W),
            bounds=bounds).x

    for i in range(iters):
        moments = exp_moments(饾泬, 蟿, lag)

        饾悹 = empMoments - moments
        S = 饾悹.dot(饾悹.T)

        W = np.linalg.inv(S)
        W /= np.max(W) # Avoid overflow of the loss function

        饾泬 = minimize(exp_gmm_loss, x0=饾泬,
                args=(蟿, lag, empMoments, W),
                bounds=bounds).x

    return 饾泬


# Fit EM


@njit(nogil=True, parallel=True)
def em_responsibilities(饾惌, 饾泬):
    位, 伪, 尾 = 饾泬

    N = len(饾惌)
    resp = np.empty((N,N), dtype=np.float64)

    for i in prange(0,N):
        if i == 0:
            resp[i, 0] = 1.0
            for j in range(1, N):
                resp[i, j] = 0.0
        else:
            resp[i, 0] = 位
            rowSum = 位

            for j in range(1, i+1):
                resp[i, j] = 伪*np.exp(-尾*(饾惌[i] - 饾惌[j-1]))
                rowSum += resp[i, j]

            for j in range(0, i+1):
                resp[i, j] /= rowSum

            for j in range(i+1, N):
                resp[i, j] = 0.0
    return resp


def exp_em(饾惌, T, 饾泬_start=np.array([1.0, 2.0, 3.0]), iters=100, verbosity=None, calcLikelihoods=False):
    """
    Run an EM fit on the '饾惌' arrival times up until final time 'T'.
    """
    饾泬 = 饾泬_start.copy()

    llIterations = np.zeros(iters)
    iters = tqdm(range(iters)) if verbosity else range(iters)

    for i in iters:
        饾泬, ll = exp_em_iter(饾惌, T, 饾泬, calcLikelihoods)
        llIterations[i] = ll

        if verbosity and i % verbosity == 0:
            print(饾泬[0], 饾泬[1], 饾泬[2])

    if calcLikelihoods:
        return 饾泬, llIterations
    else:
        return 饾泬


@njit(nogil=True, parallel=True)
def exp_em_iter(饾惌, T, 饾泬, calcLikelihoods):
    位, 伪, 尾 = 饾泬
    N = len(饾惌)

    # E step
    resp = em_responsibilities(饾惌, 饾泬)

    # M step: Update 位
    位 = np.sum(resp[:,0])/T

    # M step: Update 伪
    numer = np.sum(resp[:,1:])
    denom = np.sum(1 - np.exp(-尾*(T - 饾惌)))
    伪 = 尾*numer/denom

    # M step: Update 尾
    numer = np.sum(1 - np.exp(-尾*(T - 饾惌)))/尾 - np.sum((T - 饾惌)*np.exp(-尾*(T - 饾惌)))

    denom = 0
    for j in prange(1, N):
        denom += np.sum((饾惌[j] - 饾惌[:j])*resp[j,1:j+1])

    尾 = 伪*numer/denom

    if calcLikelihoods:
        ll = exp_log_likelihood(饾惌, T, 饾泬)
    else:
        ll = 0.0

    饾泬[0] = 位
    饾泬[1] = 伪
    饾泬[2] = 尾

    return 饾泬, ll


## Mutually exciting Hawkes with exponential decay
@njit()
def mutual_hawkes_intensity(t, 鈩媉t, 饾泬):
    """
    Each 渭[i] is an m-vector-valued function, which takes as argument
    the time passed since an arrival to process i, and returns the
    lasting effect on each of the m processes
    """
    位, 渭 = 饾泬

    位耍 = 位
    for (t_i, d_i) in 鈩媉t:
        位耍 += 渭[d_i](t - t_i)
    return 位耍


@njit(nogil=True)
def mutual_exp_hawkes_intensity(t, times, ids, 饾泬):
    """
    The 位 is an m-vector which shows the starting intensity for
    each process.

    Each 伪[i] is an m-vector which shows the jump in intensity
    for each of the processes when an arrival comes to process i.

    The 尾 is an m-vector which shows the intensity decay rates for
    each processes intensity.
    """
    位, 伪, 尾 = 饾泬

    位耍 = 位.copy()
    for (t_i, d_i) in zip(times, ids):
        位耍 += 伪[d_i] * np.exp(-尾 * (t - t_i))

    return 位耍


@njit(nogil=True)
def mutual_exp_hawkes_compensator(t, times, ids, 饾泬):
    """
    The 位 is an m-vector which shows the starting intensity for
    each process.

    Each 伪[i] is an m-vector which shows the jump in intensity
    for each of the processes when an arrival comes to process i.

    The 尾 is an m-vector which shows the intensity decay rates for
    each processes intensity.
    """
    # if t <= 0: return np.zeros(m)

    位, 伪, 尾 = 饾泬

    螞 = 位 * t

    for (t_i, d_i) in zip(times, ids):
        # 螞 += M(t - t_i, d_i)
        螞 += (伪[d_i]/尾) * (1 - np.exp(-尾*(t - t_i)))
    return 螞


@njit(nogil=True)
def mutual_exp_hawkes_compensators(times, ids, 饾泬):
    """
    The 位 is an m-vector which shows the starting intensity for
    each process.

    Each 伪[i] is an m-vector which shows the jump in intensity
    for each of the processes when an arrival comes to process i.

    The 尾 is an m-vector which shows the intensity decay rates for
    each processes intensity.
    """

    位, 伪, 尾 = 饾泬
    m = len(位)

    螞 = np.zeros(m)
    位耍_prev = 位
    t_prev = 0

    螞s = np.zeros((len(times), m), dtype=np.float64)

    for i in range(len(times)):
        t_i = times[i]
        d_i = ids[i]

        螞 += 位 * (t_i - t_prev) + (位耍_prev - 位)/尾 * (1 - np.exp(-尾*(t_i - t_prev)))
        螞s[i,:] = 螞

        位耍_prev = 位 + (位耍_prev - 位) * np.exp(-尾 * (t_i - t_prev)) + 伪[d_i,:]
        t_prev = t_i

    return 螞s


@njit(nogil=True)
def mutual_log_likelihood(鈩媉T, T, 饾泬, 位耍, 螞):
    m = len(饾泬)
    鈩? = 0
    for (t_i, d_i) in 鈩媉T:
        if t_i > T:
            raise RuntimeError("T is too small for this data")

        # Get the history of arrivals before time t_i
        鈩媉i = [(t_s, d_s) for (t_s, d_s) in 鈩媉T if t_s < t_i]
        位耍_i = 位耍(t_i, 鈩媉i, 饾泬)
        鈩? += np.log(位耍_i[d_i])

    鈩? -= np.sum(螞(T, 鈩媉T, 饾泬))
    return 鈩?


@njit(nogil=True)
def mutual_exp_log_likelihood(times, ids, T, 饾泬):
    if np.max(times) > T:
        raise RuntimeError("T is too small for this data")

    位, 伪, 尾 = 饾泬

    if np.min(位) <= 0 or np.min(伪) < 0 or np.min(尾) <= 0: return -np.inf

    鈩? = 0
    位耍 = 饾泬[0]

    t_prev = 0
    for t_i, d_i in zip(times, ids):
        位耍 = 位 + (位耍 - 位) * np.exp(-尾 * (t_i - t_prev))
        鈩? += np.log(位耍[d_i])

        位耍 += 伪[d_i,:]
        t_prev = t_i

    鈩? -= np.sum(mutual_exp_hawkes_compensator(T, times, ids, 饾泬))

    return 鈩?


def mutual_exp_simulate_by_thinning(饾泬, T):

    """
    The 位 is an m-vector which shows the starting intensity for
    each process.

    Each 伪[i] is an m-vector which shows the jump in intensity
    for each of the processes when an arrival comes to process i.

    The 尾 is an m-vector which shows the intensity decay rates for
    each processes intensity.
    """
    位, 伪, 尾 = 饾泬
    m = len(位)

    位耍 = 位
    times = []

    t = 0

    while True:
        M = np.sum(位耍)
        螖t = rnd.exponential() / M
        t += 螖t
        if t > T:
            break

        位耍 = 位 + (位耍 - 位) * np.exp(-尾 * 螖t)

        u = M * rnd.rand()
        if u > np.sum(位耍):
            continue # No arrivals (they are 'thinned' out)

        cumulative位耍 = 0

        for i in range(m):
            cumulative位耍 += 位耍[i]
            if u < cumulative位耍:
                times.append((t, i))
                位耍 += 伪[i]
                break

    return times


def flatten_theta(饾泬):
    return np.hstack([饾泬[0], np.hstack(饾泬[1]), 饾泬[2]])


def unflatten_theta(饾泬_flat, m):
    位 = 饾泬_flat[:m]
    伪 = 饾泬_flat[m:(m + m**2)].reshape((m,m))
    尾 = 饾泬_flat[(m + m**2):]

    return (位, 伪, 尾)


def mutual_exp_mle(饾惌, ids, T, 饾泬_start):

    m = len(饾泬_start[0])
    饾泬_start_flat = flatten_theta(饾泬_start)

    def loss(饾泬_flat):
        return -mutual_exp_log_likelihood(饾惌, ids, T, unflatten_theta(饾泬_flat, m))

    def print_progress(饾泬_i, itCount = []):
        itCount.append(None)
        i = len(itCount)

        if i % 100 == 0:
            ll = -loss(饾泬_i)
            print(f"Iteration {i} loglikelihood {ll:.2f}")

    res = minimize(loss, 饾泬_start_flat, options={"disp": True, "maxiter": 100_000},
        callback = print_progress, method = 'Nelder-Mead')

    饾泬_mle = unflatten_theta(res.x, m)
    logLike = -res.fun

    return 饾泬_mle, logLike


# More advanced MLE methods for the exponential case


@njit()
def ozaki_recursion(饾惌, 饾泬, n):
    """
    Calculate sum_{j=1}^{i-1} t_j^n * exp(-尾 * (t_i - t_j)) recursively
    """
    位, 伪, 尾 = 饾泬
    N_T = len(饾惌)

    A_n = np.empty(N_T, dtype=np.float64)
    A_n[0] = 0
    for i in range(1, N_T):
        A_n[i] = np.exp(-尾*(饾惌[i] - 饾惌[i-1])) * (饾惌[i-1]**n + A_n[i-1])

    return A_n


@njit()
def deriv_exp_log_likelihood(鈩媉T, T, 饾泬):
    位, 伪, 尾 = 饾泬

    饾惌 = 鈩媉T
    N_T = len(饾惌)

    A = ozaki_recursion(饾惌, 饾泬, 0)
    A_1 = ozaki_recursion(饾惌, 饾泬, 1)

    B = np.empty(N_T, dtype=np.float64)
    B[0] = 0

    for i in range(1, N_T):
        B[i] = 饾惌[i] * A[i] - A_1[i]

    d鈩揹位 = -T
    d鈩揹伪 = 0
    d鈩揹尾 = 0

    for i, t_i in enumerate(鈩媉T):
        d鈩揹伪 += (1/尾) * (np.exp(-尾*(T-t_i)) - 1) + A[i] / (位 + 伪 * A[i])
        d鈩揹尾 += -伪 * ( (1/尾) * (T - t_i) * np.exp(-尾*(T-t_i)) \
                     + (1/尾**2) * (np.exp(-尾*(T-t_i))-1) ) \
                - (伪 * B[i] / (位 + 伪 * A[i]))
        d鈩揹位 += 1 / (位 + 伪 * A[i])

    d = np.empty(3, dtype=np.float64)
    d[0] = d鈩揹位
    d[1] = d鈩揹伪
    d[2] = d鈩揹尾
    return d


@njit()
def hess_exp_log_likelihood(鈩媉T, T, 饾泬):
    位, 伪, 尾 = 饾泬

    饾惌 = 鈩媉T
    N_T = len(饾惌)

    A = ozaki_recursion(饾惌, 饾泬, 0)
    A_1 = ozaki_recursion(饾惌, 饾泬, 1)
    A_2 = ozaki_recursion(饾惌, 饾泬, 2)

    # B is sum (t_i - t_j) * exp(- ...)
    # C is sum (t_i - t_j)**2 * exp(- ...)
    B = np.empty(N_T, dtype=np.float64)
    C = np.empty(N_T, dtype=np.float64)
    B[0] = 0
    C[0] = 0

    for i in range(1, N_T):
        B[i] = 饾惌[i] * A[i] - A_1[i]
        C[i] = 饾惌[i]**2 * A[i] - 2*饾惌[i]*A_1[i] + A_2[i]

    d2鈩揹伪2 = 0
    d2鈩揹伪d尾 = 0
    d2鈩揹尾2 = 0

    d2鈩揹位2 = 0
    d2鈩揹伪d位 = 0
    d2鈩揹尾d位 = 0

    for i, t_i in enumerate(鈩媉T):
        d2鈩揹伪2 += - ( A[i] / (位 + 伪 * A[i]) )**2
        d2鈩揹伪d尾 += - ( (1/尾) * (T - t_i) * np.exp(-尾*(T-t_i)) \
                     + (1/尾**2) * (np.exp(-尾*(T-t_i))-1) ) \
                   + ( -B[i]/(位 + 伪 * A[i]) + (伪 * A[i] * B[i]) / (位 + 伪 * A[i])**2 )

        d2鈩揹尾2 += 伪 * ( (1/尾) * (T - t_i)**2 * np.exp(-尾*(T-t_i)) + \
                        (2/尾**2) * (T - t_i) * np.exp(-尾*(T-t_i)) + \
                        (2/尾**3) * (np.exp(-尾*(T-t_i)) - 1) ) + \
                  ( 伪*C[i] / (位 + 伪 * A[i]) - (伪*B[i] / (位 + 伪 * A[i]))**2 )


        d2鈩揹位2 += -1 / (位 + 伪 * A[i])**2
        d2鈩揹伪d位 += -A[i] / (位 + 伪 * A[i])**2
        d2鈩揹尾d位 += 伪 * B[i] / (位 + 伪 * A[i])**2

    H = np.empty((3,3), dtype=np.float64)
    H[0,0] = d2鈩揹位2
    H[1,1] = d2鈩揹伪2
    H[2,2] = d2鈩揹尾2
    H[0,1] = H[1,0] = d2鈩揹伪d位
    H[0,2] = H[2,0] = d2鈩揹尾d位
    H[1,2] = H[2,1] = d2鈩揹伪d尾
    return H


def exp_mle_with_grad(饾惌, T, 饾泬_start=np.array([1.0, 2.0, 3.0])):
    eps = 1e-5
    饾泬_bounds = ((eps, None), (eps, None), (eps, None))
    loss = lambda 饾泬: -exp_log_likelihood(饾惌, T, 饾泬)
    grad = lambda 饾泬: -deriv_exp_log_likelihood(饾惌, T, 饾泬)
    饾泬_mle = minimize(loss, 饾泬_start, bounds=饾泬_bounds, jac=grad).x

    return 饾泬_mle


def exp_mle_with_hess(饾惌, T, 饾泬_start=np.array([1.0, 2.0, 3.0])):
    eps = 1e-5
    饾泬_bounds = ((eps, None), (eps, None), (eps, None))
    loss = lambda 饾泬: -exp_log_likelihood(饾惌, T, 饾泬)
    grad = lambda 饾泬: -deriv_exp_log_likelihood(饾惌, T, 饾泬)
    hess = lambda 饾泬: -hess_exp_log_likelihood(饾惌, T, 饾泬)
    饾泬_mle = minimize(loss, 饾泬_start, bounds=饾泬_bounds, jac=grad, hess=hess,
        method="trust-constr").x

    return 饾泬_mle


# Alternative simulation method


@njit(nogil=True)
def exp_simulate_by_composition_alt(饾泬, T):
    """
    This is simply an alternative to 'exp_simulate_by_composition'
    where the simulation stops after time T rather than stopping after
    observing N arrivals.
    """
    位, 伪, 尾 = 饾泬
    位耍_k = 位
    t_k = 0

    鈩? = []
    while t_k < T:
        U_1 = rnd.rand()
        U_2 = rnd.rand()

        # Technically the following works, but without @njit
        # it will print out "RuntimeWarning: invalid value encountered in log".
        # This is because 1 + 尾/(位耍_k + 伪 - 位)*np.log(U_2) can be negative
        # so T_2 can be np.NaN. The Dassios & Zhao (2013) algorithm checks if this
        # expression is negative and handles it separately, though the lines
        # below have the same behaviour as t_k = min(T_1, np.NaN) will be T_1. 
        T_1 = t_k - np.log(U_1) / 位
        T_2 = t_k - np.log(1 + 尾/(位耍_k + 伪 - 位)*np.log(U_2))/尾

        t_prev = t_k
        t_k = min(T_1, T_2)
        鈩?.append(t_k)

        if len(鈩?) > 1:
            位耍_k = 位 + (位耍_k + 伪 - 位) * (
                    np.exp(-尾 * (t_k - t_prev)))
        else:
            位耍_k = 位

    return np.array(鈩媅:-1])
