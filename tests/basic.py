import hawkesbook as hawkes

import numpy as np
import numpy.random as rnd
from tqdm import tqdm

from numpy.testing import assert_allclose

empMean, empVar, empAutoCov = hawkes.empirical_moments([1, 2, 2.1, 2.3, 4.5, 9.9], T=10, τ=2, lag=1)
assert min(empMean, empVar) > 0

assert hawkes.hawkes_intensity(1, [], [1, None, None]) == 1
assert hawkes.hawkes_intensity(2, [1], [1, lambda x: np.exp(-x), None]) == 1 + np.exp(-1)

assert hawkes.exp_hawkes_intensity(1, [0.5], [1.0, 2.0, 3.0]) == hawkes.hawkes_intensity(1, [0.5], (1, lambda t: 2*np.exp(-3*t), None))

testα = 3
testβ = 4
testμ = lambda x: testα*np.exp(-testβ * x)
testM = lambda t: (testα/testβ) * (1 - np.exp(-testβ*t))

testα = testβ = 1
testμ = lambda x: testα*np.exp(-testβ * x)
testM = lambda t: (testα/testβ) * (1 - np.exp(-testβ*t))

rnd.seed(1)
simTimes = hawkes.simulate_inverse_compensator([1, testμ, testM], hawkes.hawkes_compensator, 10)
print(simTimes)

print("Testing log likelihoods")
testObs = np.array([0.5, 0.75])
testT = 1.0
test𝛉 = np.array([1.0, 2.0, 3.0])
assert hawkes.exp_log_likelihood(testObs, testT, test𝛉) == hawkes.log_likelihood(testObs, testT, test𝛉,
    hawkes.exp_hawkes_intensity, hawkes.exp_hawkes_compensator)

assert_allclose(hawkes.exp_log_likelihood(testObs, testT, test𝛉),
               hawkes.log_likelihood(testObs, testT, test𝛉, hawkes.exp_hawkes_intensity, hawkes.exp_hawkes_compensator),
               0.1)

print(f"Passed! Exp version = {hawkes.exp_log_likelihood(testObs, testT, test𝛉)} == general verison = {hawkes.log_likelihood(testObs, testT, test𝛉, hawkes.exp_hawkes_intensity, hawkes.exp_hawkes_compensator)}")

# Test simulation methods for exponential case
sim𝛉 = np.array([1.0, 2.0, 3.0])
testT = 100

hawkes.numba_seed(1)
N_T = []
max_t = []
for r in tqdm(range(10_000)):
    times = hawkes.exp_simulate_by_composition(sim𝛉, 1_000)
    N_T.append(len(times[times < testT]))
    max_t.append(times[-1])
print(f"Over [0, 100] we had {np.mean(N_T)} arrivals on average by composition method")

hawkes.numba_seed(1)
N_T = []
for r in tqdm(range(10_000)):
    N_T.append(len(hawkes.exp_simulate_by_composition_alt(sim𝛉, testT)))
print(f"Over [0, 100] we had {np.mean(N_T)} arrivals on average by alternative composition method")


hawkes.numba_seed(1)
N_T = []
for r in tqdm(range(10_000)):
    N_T.append(len(hawkes.exp_simulate_by_thinning(sim𝛉, testT)))
print(f"Over [0, 100] we had {np.mean(N_T)} arrivals on average by thinning method")

testλ, testα, testβ = sim𝛉
testμ = lambda x: testα*np.exp(-testβ * x)
testM = lambda t: (testα/testβ) * (1 - np.exp(-testβ*t))

rnd.seed(1)
N_T = []
max_t = []
for r in tqdm(range(10)):
    times = hawkes.simulate_inverse_compensator([testλ, testμ, testM], hawkes.hawkes_compensator, 500)
    N_T.append(len(times[times < testT]))
    max_t.append(times[-1])
print(f"Over [0, 100] we had {np.mean(N_T)} arrivals on average by inverse compensator method")

hawkes.numba_seed(1)

sim𝛉 = np.array([1.0, 2.0, 3.1])

testT = 1_000
testObs = hawkes.exp_simulate_by_thinning(sim𝛉, testT)
print(f"Testing log likelihoods on larger sample (of size {len(testObs)})")

print(f"Exp version = {hawkes.exp_log_likelihood(testObs, testT, test𝛉)} == general verison = {hawkes.log_likelihood(testObs, testT, test𝛉, hawkes.exp_hawkes_intensity, hawkes.exp_hawkes_compensator)}")

assert_allclose(hawkes.exp_log_likelihood(testObs, testT, test𝛉),
               hawkes.log_likelihood(testObs, testT, test𝛉, hawkes.exp_hawkes_intensity, hawkes.exp_hawkes_compensator),
               0.1)
print("Passed!")

testMean, testVar, testAutoCov = testMoments = hawkes.exp_moments([1, 2, 3], τ=2, lag=1)
assert min(testMean, testVar) > 0

assert hawkes.exp_gmm_loss(np.array([1.0, 2.0, 3.0]), 2, 1, testMoments + 2, np.eye(3)) == 12.0

print(hawkes.exp_gmm(np.array([1.0, 1.1, 1.2, 5.0]), 10, 2, 1))

print("Testing EM algorithm")
print(hawkes.exp_em(np.array([1.0, 1.1, 1.2, 5.0]), 6.0, np.array([1.0, 2.0, 3.0]), 10, 1, True))

fit = hawkes.exp_mle(np.array([1.0, 1.1, 1.2, 5.0]), 10)