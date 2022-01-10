# _hawkesbook_ Python package for Hawkes Process inference, simulation, etc.

This package implements inference, simulation, and other related method for Hawkes processes and some mutually-exciting Hawkes processes.
It is meant to accompany [_The Elements of Hawkes Processes_](https://link.springer.com/book/10.1007/978-3-030-84639-8) written by Patrick J. Laub, Young Lee, and Thomas Taimre.
To install simply run `pip install hawkesbook`.

The main design goal for this package was simplicity and readability.
Unicode characters are used to match the mathematics to the code as much as possible.
For example, the Hawkes process conditional intensity (in LaTeX notation) is
```latex
\lambda^\ast(t) = \lambda + \sum_{t_i \in \mathcal{H}_t} \mu(t - t_i)
```
and this is rendered in Python as
```python
def hawkes_intensity(t, ℋ_t, 𝛉):
    λ, μ, _ = 𝛉
    λˣ = λ
    for t_i in ℋ_t:
        λˣ += μ(t - t_i)
    return λˣ
```

Some functions are JIT-compiled to C and parallelised with `numba` so the computational performance is not completely neglected.
Everything that can be `numpy`-vectorised has been.

Our main dependencies are `numba`, `numpy`, and `scipy` (for the minimize function).

As an example, in the book we have a case study which fits various Hawkes process to the arrival times of earthquakes.
The code for the fitting and analysis of that data is like:

```python
import hawkesbook as hawkes

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot

# Load data to fit
quakes = pd.read_csv("japanese-earthquakes.csv")
quakes.index = pd.to_datetime(quakes.Day.astype(str) + "/" + quakes.Month.astype(str) + "/" + quakes.Year.astype(str) + " " + quakes.Time, dayfirst=True)
quakes.sort_index(inplace=True)

# Calculate each arrival as a (fractional) number of days since the
# beginning of the observation period
timeToQuake = quakes.index - pd.Timestamp("1/1/1973")
ts = np.array(timeToQuake.total_seconds() / 60 / 60 / 24)

# Calculate the length of the observation period
obsPeriod = pd.Timestamp("31/12/2020") - pd.Timestamp("1/1/1973")
T = obsPeriod.days

# Calculate the maximum likelihood estimate for the Hawkes process
# with an exponentially decaying intensity
𝛉_exp_mle = hawkes.exp_mle(ts, T)
print("Exp Hawkes MLE fit: ", 𝛉_exp_mle)

# Calculate the EM estimate or the same type of Hawkes process
𝛉_exp_em = hawkes.exp_em(ts, T, iters=100)
print("Exp Hawkes EM fit: ", 𝛉_exp_mle)

# Get the likelihoods of each fit to find the better one
ll_mle = hawkes.exp_log_likelihood(ts, T, 𝛉_exp_mle)
ll_em = hawkes.exp_log_likelihood(ts, T, 𝛉_exp_em)

if ll_mle > ll_em:
	print("MLE was a better fit than EM in this case")
	𝛉_exp = 𝛉_exp_mle
	ll_exp = ll_mle
else:
	print("EM was a better fit than MLE in this case")
	𝛉_exp = 𝛉_exp_em
	ll_exp = ll_em

# Fit instead the Hawkes with a power-law decay
𝛉_pl = hawkes.power_mle(ts, T)
ll_pl = hawkes.power_log_likelihood(ts, T, 𝛉_pl)

# Compare the BICs
BIC_exp = 3 * np.log(len(ts)) - 2 * ll_exp
BIC_pl = 4 * np.log(len(ts)) - 2 * ll_pl
if BIC_exp < BIC_pl:
	print(f"The exponentially-decaying Hawkes was the better fit with BIC={BIC_exp:.2f}.")
	print(f"The power-law Hawkes had BIC={BIC_pl:.2f}.")
else:
	print(f"The power-law Hawkes was the better fit with BIC={BIC_pl:.2f}.")
	print(f"The exponentially-decaying Hawkes had BIC={BIC_exp:.2f}.")

# Create a Q-Q plot for the exponential-decay fit by
# first transforming the points to a unit-rate Poisson
# process as outlined by the random time change theorem
tsShifted = hawkes.exp_hawkes_compensators(ts, 𝛉_exp)
iat = np.diff(np.insert(tsShifted, 0, 0))
qqplot(iat, dist=stats.expon, fit=False, line="45")
plt.show()
```
