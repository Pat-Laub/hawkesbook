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


def hawkes_intensity(t, ℋ_t, 𝛉):
    λ, μ, _ = 𝛉
    λˣ = λ
    for t_i in ℋ_t:
        λˣ += μ(t - t_i)
    return λˣ


def hawkes_compensator(t, ℋ_t, 𝛉):
    if t <= 0: return 0
    λ, _, M = 𝛉

    Λ = λ * t
    for t_i in ℋ_t:
        Λ += M(t - t_i)
    return Λ


def exp_hawkes_intensity(t, ℋ_t, 𝛉):
    λ, α, β = 𝛉
    λˣ = λ
    for t_i in ℋ_t:
        λˣ += α * np.exp(-β * (t - t_i))
    return λˣ


def exp_hawkes_compensator(t, ℋ_t, 𝛉):
    if t <= 0: return 0
    λ, α, β = 𝛉
    Λ = λ * t
    for t_i in ℋ_t:
        Λ += (α/β) * (1 - np.exp(-β*(t - t_i)))
    return Λ


@njit(nogil=True)
def exp_hawkes_compensators(ℋ_t, 𝛉):
    λ, α, β = 𝛉

    Λ = 0
    λˣ_prev = λ
    t_prev = 0

    Λs = np.empty(len(ℋ_t), dtype=np.float64)
    for i, t_i in enumerate(ℋ_t):
        Λ += λ * (t_i - t_prev) + (
                (λˣ_prev - λ)/β *
                (1 - np.exp(-β*(t_i - t_prev))))
        Λs[i] = Λ

        λˣ_prev = λ + (λˣ_prev - λ) * (
                np.exp(-β * (t_i - t_prev))) + α
        t_prev = t_i
    return Λs


@njit(nogil=True)
def power_hawkes_intensity(t, ℋ_t, 𝛉):
    λ, k, c, p = 𝛉
    λˣ = λ
    for t_i in ℋ_t:
        λˣ += k / (c + (t-t_i))**p
    return λˣ


@njit(nogil=True)
def power_hawkes_compensator(t, ℋ_t, 𝛉):
    λ, k, c, p = 𝛉
    Λ = λ * t
    for t_i in ℋ_t:
        Λ += ((k * (c * (c + (t-t_i)))**-p *
              (-c**p * (c + (t-t_i)) + c * (c + (t-t_i))**p)) /
              (p - 1))
    return Λ


@njit(nogil=True, parallel=True)
def power_hawkes_compensators(ℋ_t, 𝛉):
    Λs = np.empty(len(ℋ_t), dtype=np.float64)
    for i in prange(len(ℋ_t)):
        t_i = ℋ_t[i]
        ℋ_i = ℋ_t[:i]
        Λs[i] = power_hawkes_compensator(t_i, ℋ_i, 𝛉)
    return Λs


# Likelihood

def log_likelihood(ℋ_T, T, 𝛉, λˣ, Λ):
    ℓ = 0.0
    for i, t_i in enumerate(ℋ_T):
        ℋ_i = ℋ_T[:i]
        λˣ_i = λˣ(t_i, ℋ_i, 𝛉)
        ℓ += np.log(λˣ_i)
    ℓ -= Λ(T, ℋ_T, 𝛉)
    return ℓ


@njit(nogil=True, parallel=True)
def power_log_likelihood(ℋ_T, T, 𝛉):
    ℓ = 0.0
    for i in prange(len(ℋ_T)):
        t_i = ℋ_T[i]
        ℋ_i = ℋ_T[:i]
        λˣ_i = power_hawkes_intensity(t_i, ℋ_i, 𝛉)
        ℓ += np.log(λˣ_i)
    ℓ -= power_hawkes_compensator(T, ℋ_T, 𝛉)
    return ℓ


@njit()
def exp_log_likelihood(ℋ_T, T, 𝛉):
    λ, α, β = 𝛉
    𝐭 = ℋ_T
    N_T = len(𝐭)

    A = np.empty(N_T, dtype=np.float64)
    A[0] = 0
    for i in range(1, N_T):
        A[i] = np.exp(-β*(𝐭[i] - 𝐭[i-1])) * (1 + A[i-1])

    ℓ = -λ*T
    for i, t_i in enumerate(ℋ_T):
        ℓ += np.log(λ + α * A[i]) - \
                (α/β) * (1 - np.exp(-β*(T-t_i)))
    return ℓ


def exp_mle(𝐭, T, 𝛉_start=np.array([1.0, 2.0, 3.0])):
    eps = 1e-5
    𝛉_bounds = ((eps, None), (eps, None), (eps, None))
    loss = lambda 𝛉: -exp_log_likelihood(𝐭, T, 𝛉)
    𝛉_mle = minimize(loss, 𝛉_start, bounds=𝛉_bounds).x
    return np.array(𝛉_mle)


def power_mle(𝐭, T, 𝛉_start=np.array([1.0, 1.0, 2.0, 3.0])):
    eps = 1e-5
    𝛉_bounds = ((eps, None), (eps, None), (eps, None),
        (1+eps, 100))
    loss = lambda 𝛉: -power_log_likelihood(𝐭, T, 𝛉)
    𝛉_mle = minimize(loss, 𝛉_start, bounds=𝛉_bounds).x
    return np.array(𝛉_mle)


# Simulation


def simulate_inverse_compensator(𝛉, Λ, N):
    ℋ = np.empty(N, dtype=np.float64)

    tˣ_1 = -np.log(rnd.rand())
    exp_1 = lambda t_1: Λ(t_1, ℋ[:0], 𝛉) - tˣ_1

    t_1_guess = 1.0
    t_1 = fsolve(exp_1, t_1_guess)[0]

    ℋ[0] = t_1
    t_prev = t_1
    for i in range(1, N):
        Δtˣ_i = -np.log(rnd.rand())

        Λ_i = Λ(t_prev, ℋ, 𝛉)
        exp_i = lambda t_next: Λ(t_next, ℋ[:i], 𝛉) - Λ_i - Δtˣ_i

        t_next_guess = t_prev + 1.0
        t_next = fsolve(exp_i, t_next_guess)[0]

        ℋ[i] = t_next
        t_prev = t_next
    return ℋ

@njit(nogil=True)
def exp_simulate_by_composition(𝛉, N):
    λ, α, β = 𝛉
    λˣ_k = λ
    t_k = 0

    ℋ = np.empty(N, dtype=np.float64)
    for k in range(N):
        U_1 = rnd.rand()
        U_2 = rnd.rand()

        T_1 = t_k - np.log(U_1) / λ
        T_2 = t_k - np.log(1 + β/(λˣ_k + α - λ)*np.log(U_2))/β

        t_prev = t_k
        t_k = min(T_1, T_2)
        ℋ[k] = t_k

        λˣ_k = λ + (λˣ_k + α - λ) * (
                np.exp(-β * (t_k - t_prev)))
    return ℋ


@njit(nogil=True)
def exp_simulate_by_thinning(𝛉, T):
    λ, α, β = 𝛉

    λˣ = λ
    times = []

    t = 0

    while True:
        M = λˣ
        Δt = rnd.exponential() / M
        t += Δt
        if t > T:
            break

        λˣ = λ + (λˣ - λ) * np.exp(-β * Δt)

        u = M * rnd.rand()
        if u > λˣ:
            continue  # This potential arrival is 'thinned' out

        times.append(t)
        λˣ += α

    return np.array(times)


@njit(nogil=True)
def power_simulate_by_thinning(𝛉, T):
    λ, k, c, p = 𝛉

    λˣ = λ
    times = []

    t = 0

    while True:
        M = λˣ
        Δt = rnd.exponential() / M
        t += Δt
        if t > T:
            break

        λˣ = power_hawkes_intensity(t, np.array(times), 𝛉)

        u = M * rnd.rand()
        if u > λˣ:
            continue  # This potential arrival is 'thinned' out

        times.append(t)
        λˣ += k / (c ** p)

    return np.array(times)


# Moment matching


def empirical_moments(𝐭, T, τ, lag):
    bins = np.arange(0, T, τ)
    N = len(bins) - 1
    count = np.zeros(N)

    for i in range(N):
        count[i] = np.sum((bins[i] <= 𝐭) & (𝐭 < bins[i+1]))

    empMean = np.mean(count)
    empVar = np.std(count)**2
    empAutoCov = np.mean((count[:-lag] - empMean) \
                    * (count[lag:] - empMean))

    return np.array([empMean, empVar, empAutoCov]).reshape(3,1)



def exp_moments(𝛉, τ, lag):
    """
    Consider an exponential Hawkes process with parameter 𝛉.
    Look at intervals of length τ, i.e. N(t+τ) - N(t).
    Calculate the limiting (t->∞) mean and variance.
    Also, get the limiting autocovariance:
        E[ (N(t + τ) - N(t)) (N(t + lag*τ + τ) - N(t + lag*τ)) ].
    """
    λ, α, β = 𝛉
    κ = β - α
    δ = lag*τ

    mean = (λ*β/κ)*τ
    var = (λ*β/κ)*(τ*(β/κ) + (1 - β/κ)*((1 - np.exp(-κ*τ))/κ))
    autoCov = (λ*β*α*(2*β-α)*(np.exp(-κ*τ) - 1)**2/(2*κ**4)) \
                *np.exp(-κ*δ)

    return np.array([mean, var, autoCov]).reshape(3,1)


def exp_gmm_loss(𝛉, τ, lag, empMoments, W):
    moments = exp_moments(𝛉, τ, lag)
    𝐠 = empMoments - moments
    return (𝐠.T).dot(W).dot(𝐠)[0,0]

def exp_gmm(𝐭, T, τ=5, lag=5, iters=2, 𝛉_start=np.array([1.0, 2.0, 3.0])):
    empMoments = empirical_moments(𝐭, T, τ, lag)

    W = np.eye(3)
    bounds = ((0, None), (0, None), (0, None))

    𝛉 = minimize(exp_gmm_loss, x0=𝛉_start,
            args=(τ, lag, empMoments, W),
            bounds=bounds).x

    for i in range(iters):
        moments = exp_moments(𝛉, τ, lag)

        𝐠 = empMoments - moments
        S = 𝐠.dot(𝐠.T)

        W = np.linalg.inv(S)
        W /= np.max(W) # Avoid overflow of the loss function

        𝛉 = minimize(exp_gmm_loss, x0=𝛉,
                args=(τ, lag, empMoments, W),
                bounds=bounds).x

    return 𝛉


# Fit EM


@njit(nogil=True, parallel=True)
def em_responsibilities(𝐭, 𝛉):
    λ, α, β = 𝛉

    N = len(𝐭)
    resp = np.empty((N,N), dtype=np.float64)

    for i in prange(0,N):
        if i == 0:
            resp[i, 0] = 1.0
            for j in range(1, N):
                resp[i, j] = 0.0
        else:
            resp[i, 0] = λ
            rowSum = λ

            for j in range(1, i+1):
                resp[i, j] = α*np.exp(-β*(𝐭[i] - 𝐭[j-1]))
                rowSum += resp[i, j]

            for j in range(0, i+1):
                resp[i, j] /= rowSum

            for j in range(i+1, N):
                resp[i, j] = 0.0
    return resp


def exp_em(𝐭, T, 𝛉_start=np.array([1.0, 2.0, 3.0]), iters=100, verbosity=None, calcLikelihoods=False):
    """
    Run an EM fit on the '𝐭' arrival times up until final time 'T'.

    hp.fit(array<float>, float) -> array<float>
    """
    𝛉 = 𝛉_start.copy()

    llIterations = np.zeros(iters)
    iters = tqdm(range(iters)) if verbosity else range(iters)

    for i in iters:
        𝛉, ll = exp_em_iter(𝐭, T, 𝛉, calcLikelihoods)
        llIterations[i] = ll

        if verbosity and i % verbosity == 0:
            print(𝛉[0], 𝛉[1], 𝛉[2])

    if calcLikelihoods:
        return 𝛉, llIterations
    else:
        return 𝛉


@njit(nogil=True, parallel=True)
def exp_em_iter(𝐭, T, 𝛉, calcLikelihoods):
    λ, α, β = 𝛉
    N = len(𝐭)

    # E step
    resp = em_responsibilities(𝐭, 𝛉)

    # M step: Update λ
    λ = np.sum(resp[:,0])/T

    # M step: Update α
    numer = np.sum(resp[:,1:])
    denom = np.sum(1 - np.exp(-β*(T - 𝐭)))
    α = β*numer/denom

    # M step: Update β
    numer = np.sum(1 - np.exp(-β*(T - 𝐭)))/β - np.sum((T - 𝐭)*np.exp(-β*(T - 𝐭)))

    denom = 0
    for j in prange(1, N):
        denom += np.sum((𝐭[j] - 𝐭[:j])*resp[j,1:j+1])

    β = α*numer/denom

    if calcLikelihoods:
        ll = exp_log_likelihood(𝐭, T, 𝛉)
    else:
        ll = 0.0

    𝛉[0] = λ
    𝛉[1] = α
    𝛉[2] = β

    return 𝛉, ll


## Mutually exciting Hawkes with exponential decay
@njit()
def mutual_hawkes_intensity(t, ℋ_t, 𝛉):
    """
    Each μ[i] is an m-vector-valued function, which takes as argument
    the time passed since an arrival to process i, and returns the
    lasting effect on each of the m processes
    """
    λ, μ = 𝛉

    λˣ = λ
    for (t_i, d_i) in ℋ_t:
        λˣ += μ[d_i](t - t_i)
    return λˣ


@njit(nogil=True)
def mutual_exp_hawkes_intensity(t, times, ids, 𝛉):
    """
    The λ is an m-vector which shows the starting intensity for
    each process.

    Each α[i] is an m-vector which shows the jump in intensity
    for each of the processes when an arrival comes to process i.

    The β is an m-vector which shows the intensity decay rates for
    each processes intensity.
    """
    λ, α, β = 𝛉

    λˣ = λ.copy()
    for (t_i, d_i) in zip(times, ids):
        λˣ += α[d_i] * np.exp(-β * (t - t_i))

    return λˣ


@njit(nogil=True)
def mutual_exp_hawkes_compensator(t, times, ids, 𝛉):
    """
    The λ is an m-vector which shows the starting intensity for
    each process.

    Each α[i] is an m-vector which shows the jump in intensity
    for each of the processes when an arrival comes to process i.

    The β is an m-vector which shows the intensity decay rates for
    each processes intensity.
    """
    # if t <= 0: return np.zeros(m)

    λ, α, β = 𝛉

    Λ = λ * t

    for (t_i, d_i) in zip(times, ids):
        # Λ += M(t - t_i, d_i)
        Λ += (α[d_i]/β) * (1 - np.exp(-β*(t - t_i)))
    return Λ


@njit(nogil=True)
def mutual_exp_hawkes_compensators(times, ids, 𝛉):
    """
    The λ is an m-vector which shows the starting intensity for
    each process.

    Each α[i] is an m-vector which shows the jump in intensity
    for each of the processes when an arrival comes to process i.

    The β is an m-vector which shows the intensity decay rates for
    each processes intensity.
    """

    λ, α, β = 𝛉
    m = len(λ)

    Λ = np.zeros(m)
    λˣ_prev = λ
    t_prev = 0

    Λs = np.zeros((len(times), m), dtype=np.float64)

    for i in range(len(times)):
        t_i = times[i]
        d_i = ids[i]

        Λ += λ * (t_i - t_prev) + (λˣ_prev - λ)/β * (1 - np.exp(-β*(t_i - t_prev)))
        Λs[i,:] = Λ

        λˣ_prev = λ + (λˣ_prev - λ) * np.exp(-β * (t_i - t_prev)) + α[d_i,:]
        t_prev = t_i

    return Λs


@njit(nogil=True)
def mutual_log_likelihood(ℋ_T, T, 𝛉, λˣ, Λ):
    m = len(𝛉)
    ℓ = 0
    for (t_i, d_i) in ℋ_T:
        if t_i > T:
            raise RuntimeError("T is too small for this data")

        # Get the history of arrivals before time t_i
        ℋ_i = [(t_s, d_s) for (t_s, d_s) in ℋ_T if t_s < t_i]
        λˣ_i = λˣ(t_i, ℋ_i, 𝛉)
        ℓ += np.log(λˣ_i[d_i])

    ℓ -= np.sum(Λ(T, ℋ_T, 𝛉))
    return ℓ


@njit(nogil=True)
def mutual_exp_log_likelihood(times, ids, T, 𝛉):
    if np.max(times) > T:
        raise RuntimeError("T is too small for this data")

    λ, α, β = 𝛉

    if np.min(λ) <= 0 or np.min(α) < 0 or np.min(β) <= 0: return -np.inf

    ℓ = 0
    λˣ = 𝛉[0]

    t_prev = 0
    for t_i, d_i in zip(times, ids):
        λˣ = λ + (λˣ - λ) * np.exp(-β * (t_i - t_prev))
        ℓ += np.log(λˣ[d_i])

        λˣ += α[d_i,:]
        t_prev = t_i

    ℓ -= np.sum(mutual_exp_hawkes_compensator(T, times, ids, 𝛉))

    return ℓ


def mutual_exp_simulate_by_thinning(𝛉, T):

    """
    The λ is an m-vector which shows the starting intensity for
    each process.

    Each α[i] is an m-vector which shows the jump in intensity
    for each of the processes when an arrival comes to process i.

    The β is an m-vector which shows the intensity decay rates for
    each processes intensity.
    """
    λ, α, β = 𝛉
    m = len(λ)

    λˣ = λ
    times = []

    t = 0

    while True:
        M = np.sum(λˣ)
        Δt = rnd.exponential() / M
        t += Δt
        if t > T:
            break

        λˣ = λ + (λˣ - λ) * np.exp(-β * Δt)

        u = M * rnd.rand()
        if u > np.sum(λˣ):
            continue # No arrivals (they are 'thinned' out)

        cumulativeλˣ = 0

        for i in range(m):
            cumulativeλˣ += λˣ[i]
            if u < cumulativeλˣ:
                times.append((t, i))
                λˣ += α[i]
                break

    return times


def flatten_theta(𝛉):
    return np.hstack([𝛉[0], np.hstack(𝛉[1]), 𝛉[2]])


def unflatten_theta(𝛉_flat, m):
    λ = 𝛉_flat[:m]
    α = 𝛉_flat[m:(m + m**2)].reshape((m,m))
    β = 𝛉_flat[(m + m**2):]

    return (λ, α, β)


def mutual_exp_mle(𝐭, ids, T, 𝛉_start):

    m = len(𝛉_start[0])
    𝛉_start_flat = flatten_theta(𝛉_start)

    def loss(𝛉_flat):
        return -mutual_exp_log_likelihood(𝐭, ids, T, unflatten_theta(𝛉_flat, m))

    def print_progress(𝛉_i, itCount = []):
        itCount.append(None)
        i = len(itCount)

        if i % 100 == 0:
            ll = -loss(𝛉_i)
            print(f"Iteration {i} loglikelihood {ll:.2f}")

    res = minimize(loss, 𝛉_start_flat, options={"disp": True, "maxiter": 100_000},
        callback = print_progress, method = 'Nelder-Mead')

    𝛉_mle = unflatten_theta(res.x, m)
    logLike = -res.fun

    return 𝛉_mle, logLike


# More advanced MLE methods for the exponential case


@njit()
def ozaki_recursion(𝐭, 𝛉, n):
    """
    Calculate sum_{j=1}^{i-1} t_j^n * exp(-β * (t_i - t_j)) recursively
    """
    λ, α, β = 𝛉
    N_T = len(𝐭)

    A_n = np.empty(N_T, dtype=np.float64)
    A_n[0] = 0
    for i in range(1, N_T):
        A_n[i] = np.exp(-β*(𝐭[i] - 𝐭[i-1])) * (𝐭[i-1]**n + A_n[i-1])

    return A_n


@njit()
def deriv_exp_log_likelihood(ℋ_T, T, 𝛉):
    λ, α, β = 𝛉

    𝐭 = ℋ_T
    N_T = len(𝐭)

    A = ozaki_recursion(𝐭, 𝛉, 0)
    A_1 = ozaki_recursion(𝐭, 𝛉, 1)

    B = np.empty(N_T, dtype=np.float64)
    B[0] = 0

    for i in range(1, N_T):
        B[i] = 𝐭[i] * A[i] - A_1[i]

    dℓdλ = -T
    dℓdα = 0
    dℓdβ = 0

    for i, t_i in enumerate(ℋ_T):
        dℓdα += (1/β) * (np.exp(-β*(T-t_i)) - 1) + A[i] / (λ + α * A[i])
        dℓdβ += -α * ( (1/β) * (T - t_i) * np.exp(-β*(T-t_i)) \
                     + (1/β**2) * (np.exp(-β*(T-t_i))-1) ) \
                - (α * B[i] / (λ + α * A[i]))
        dℓdλ += 1 / (λ + α * A[i])

    d = np.empty(3, dtype=np.float64)
    d[0] = dℓdλ
    d[1] = dℓdα
    d[2] = dℓdβ
    return d


@njit()
def hess_exp_log_likelihood(ℋ_T, T, 𝛉):
    λ, α, β = 𝛉

    𝐭 = ℋ_T
    N_T = len(𝐭)

    A = ozaki_recursion(𝐭, 𝛉, 0)
    A_1 = ozaki_recursion(𝐭, 𝛉, 1)
    A_2 = ozaki_recursion(𝐭, 𝛉, 2)

    # B is sum (t_i - t_j) * exp(- ...)
    # C is sum (t_i - t_j)**2 * exp(- ...)
    B = np.empty(N_T, dtype=np.float64)
    C = np.empty(N_T, dtype=np.float64)
    B[0] = 0
    C[0] = 0

    for i in range(1, N_T):
        B[i] = 𝐭[i] * A[i] - A_1[i]
        C[i] = 𝐭[i]**2 * A[i] - 2*𝐭[i]*A_1[i] + A_2[i]

    d2ℓdα2 = 0
    d2ℓdαdβ = 0
    d2ℓdβ2 = 0

    d2ℓdλ2 = 0
    d2ℓdαdλ = 0
    d2ℓdβdλ = 0

    for i, t_i in enumerate(ℋ_T):
        d2ℓdα2 += - ( A[i] / (λ + α * A[i]) )**2
        d2ℓdαdβ += - ( (1/β) * (T - t_i) * np.exp(-β*(T-t_i)) \
                     + (1/β**2) * (np.exp(-β*(T-t_i))-1) ) \
                   + ( -B[i]/(λ + α * A[i]) + (α * A[i] * B[i]) / (λ + α * A[i])**2 )

        d2ℓdβ2 += α * ( (1/β) * (T - t_i)**2 * np.exp(-β*(T-t_i)) + \
                        (2/β**2) * (T - t_i) * np.exp(-β*(T-t_i)) + \
                        (2/β**3) * (np.exp(-β*(T-t_i)) - 1) ) + \
                  ( α*C[i] / (λ + α * A[i]) - (α*B[i] / (λ + α * A[i]))**2 )


        d2ℓdλ2 += -1 / (λ + α * A[i])**2
        d2ℓdαdλ += -A[i] / (λ + α * A[i])**2
        d2ℓdβdλ += α * B[i] / (λ + α * A[i])**2

    H = np.empty((3,3), dtype=np.float64)
    H[0,0] = d2ℓdλ2
    H[1,1] = d2ℓdα2
    H[2,2] = d2ℓdβ2
    H[0,1] = H[1,0] = d2ℓdαdλ
    H[0,2] = H[2,0] = d2ℓdβdλ
    H[1,2] = H[2,1] = d2ℓdαdβ
    return H


def exp_mle_with_grad(𝐭, T, 𝛉_start=np.array([1.0, 2.0, 3.0])):
    eps = 1e-5
    𝛉_bounds = ((eps, None), (eps, None), (eps, None))
    loss = lambda 𝛉: -exp_log_likelihood(𝐭, T, 𝛉)
    grad = lambda 𝛉: -deriv_exp_log_likelihood(𝐭, T, 𝛉)
    𝛉_mle = minimize(loss, 𝛉_start, bounds=𝛉_bounds, jac=grad).x

    return 𝛉_mle


def exp_mle_with_hess(𝐭, T, 𝛉_start=np.array([1.0, 2.0, 3.0])):
    eps = 1e-5
    𝛉_bounds = ((eps, None), (eps, None), (eps, None))
    loss = lambda 𝛉: -exp_log_likelihood(𝐭, T, 𝛉)
    grad = lambda 𝛉: -deriv_exp_log_likelihood(𝐭, T, 𝛉)
    hess = lambda 𝛉: -hess_exp_log_likelihood(𝐭, T, 𝛉)
    𝛉_mle = minimize(loss, 𝛉_start, bounds=𝛉_bounds, jac=grad, hess=hess,
        method="trust-constr").x

    return 𝛉_mle


# Alternative simulation method


@njit(nogil=True)
def exp_simulate_by_composition_alt(𝛉, T):
    λ, α, β = 𝛉
    λˣ_k = λ
    t_k = 0

    ℋ = []
    while t_k < T:
        U_1 = rnd.rand()
        U_2 = rnd.rand()

        T_1 = t_k - np.log(U_1) / λ
        T_2 = t_k - np.log(1 + β/(λˣ_k + α - λ)*np.log(U_2))/β

        t_prev = t_k
        t_k = min(T_1, T_2)
        ℋ.append(t_k)

        λˣ_k = λ + (λˣ_k + α - λ) * (
                np.exp(-β * (t_k - t_prev)))

    return np.array(ℋ[:-1])
