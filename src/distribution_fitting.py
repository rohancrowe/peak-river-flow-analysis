from pathlib import Path
import pandas as pd
import numpy as np
import scipy
from scipy import stats

## First we retrieve the processed data

# This isolates the flood-risk-analysis folder saved locally
project_root = Path(__file__).resolve().parent.parent

# Path to the locally saved flood-risk-analysis folder
csv_path = project_root / "data" / "processed" / "independent_peaks.csv"

processed_data = pd.read_csv(csv_path)
threshold_path = project_root / "data" / "processed" / "POT_threshold.csv"
POT_threshold = pd.read_csv(threshold_path)["POT_threshold"].iloc[0]

peaks_over_threshold = processed_data.copy()
peaks_over_threshold["peaks_over_threshold"] = peaks_over_threshold["peak_flow"] - POT_threshold

## Fitting the model
# The method below uses numerical approximation to estimate the MLEs of the shape (ksi) and scale (sigma) parameter by minimising the log-likelihood given the threshold (loc) = 0

ksi_hat, mu, sigma_hat = scipy.stats.genpareto.fit(peaks_over_threshold["peaks_over_threshold"], floc=0)

## Statistical testing 
# We can't use a simple Kolmogrov-Smirnov (KS) test as we'd be testing the model on the same data we fit it to
# We use bootstrapping:
# 1. Calculate the KS value for the real data
# 2. Simulate many fake datasets from the fitted GPD
# 3. Fit a GPD to each fake dataset
# 4. Calculate all the KS values for the fake datasets
# 5. Compare

from scipy.stats import genpareto

def gpd_bootstrap_test(peaks_over_threshold_data, n_boot=1000, random_state=None):
    rng = np.random.default_rng(random_state) # creates an rng for the seed from which the bootstrap simulations stems
    
    if len(peaks_over_threshold_data) < 5:
        raise ValueError("Need more data for a meaningful goodness-of-fit test.")
    
    # Observed KS statistic
    sorted_data = np.sort(np.asarray(peaks_over_threshold_data))
    n = len(sorted_data)
    empirical_cdf = np.arange(1, n + 1) / n # what proportion of values in the dataset are below the kth ordered value say
    model_cdf = genpareto.cdf(sorted_data, c=ksi_hat, loc=0, scale=sigma_hat)
    ks_obs = np.max(np.abs(empirical_cdf - model_cdf))
    
    ks_boot = np.empty(n_boot) # array of bootstrap KS statistics to be filled
    
    for b in range(n_boot):
        sim = genpareto.rvs(c=ksi_hat, loc=0, scale=sigma_hat, size=n, random_state=rng) # simulate from the fitted GPD
        sim = np.sort(sim) # sorts these exceedances by size
        
        # Refit to bootstrap sample
        try:
            shape_b, loc_b, scale_b = genpareto.fit(sim, floc=0) # fit a bootstrap GPD to the simulated data
        except Exception: # if we fail, we register nan and move on
            ks_boot[b] = np.nan
            continue
        
        empirical_cdf_b = np.arange(1, n + 1) / n
        model_cdf_b = genpareto.cdf(sim, c=shape_b, loc=0, scale=scale_b)
        ks_boot[b] = np.max(np.abs(empirical_cdf_b - model_cdf_b))
    
    ks_boot = ks_boot[np.isfinite(ks_boot)] # this drops any nan values from possible failures earlier
    p_value = np.mean(ks_boot >= ks_obs) # how often did the bootstrap KS exceed the original KS
    
    return {"shape": ksi_hat, "scale": sigma_hat, "ks_stat": ks_obs, "p_value": p_value, "bootstrap_stats": ks_boot,}


peaks_over_threshold_data = peaks_over_threshold["peaks_over_threshold"].to_numpy()

result = gpd_bootstrap_test(peaks_over_threshold_data, random_state = 10) # p-value of 0.293 indicates there is no strong evidence against the parameters ksi_hat and sigma_hat

print("shape:", result["shape"])
print("scale:", result["scale"])
print("KS statistic:", result["ks_stat"])
print("bootstrap p-value:", result["p_value"])

data = np.sort(peaks_over_threshold_data)
n = len(data)

prob = (np.arange(1, n + 1) - 0.5) / n
theoretical = genpareto.ppf(prob, c=ksi_hat, loc=0, scale=sigma_hat) # .ppf is the inverse cdf

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize = (2.7,2.7), dpi=300)

ax.scatter(theoretical, data, marker="x", s=12, linewidths=0.5, color = 'r')

max_axis = max(theoretical.max(), data.max())

ax.plot([0, max_axis], [0, max_axis], "--", linewidth=1.0, label="y = x", color = 'g')

ax.set_xlim(0, max_axis)
ax.set_ylim(0, max_axis)

from matplotlib.ticker import AutoMinorLocator

ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

ax.grid(True, which="major", alpha=0.3)
ax.grid(True, which="minor", alpha=0.15, linestyle=":")

ax.set_xlabel("Theoretical quantiles", fontsize = 6)
ax.set_ylabel("Observed quantiles", fontsize = 6)
ax.set_title("Q–Q Plot of GPD", fontsize = 8)

ax.tick_params(axis="both", labelsize=4)
fig.tight_layout()

# plt.savefig(project_root / "results" / "figures" /"QQ_plot", dpi=300, bbox_inches="tight")
# plt.show()

