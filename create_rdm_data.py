"""Script for creating offline data for the racing diffusion model.
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import polars as pl
import polars.selectors as cs
import numpy as np

import bayesflow as bf

from scipy import stats


RNG = np.random.default_rng(2024)


def truncated_normal_rvs(
    loc: float,
    scale: float,
    lower: float = 0.0,
    size: int = 1,
    random_state: int = None,
) -> np.ndarray:
    quantile_l = stats.norm.cdf(lower, loc=loc, scale=scale)

    if random_state is not None:
        probs = random_state.uniform(quantile_l, 1.0, size=size)
    else:
        probs = np.random.default_rng().uniform(quantile_l, 1.0, size=size)

    return stats.norm.ppf(
        probs,
        loc=loc,
        scale=scale,
    )


def prior():
    drift_intercept = truncated_normal_rvs(
        1, 0.5, random_state=RNG
    )
    drift_slope = truncated_normal_rvs(
        1.5, 0.5, random_state=RNG
    )
    sd_true = RNG.gamma(
        shape=12, scale=0.1
    )
    threshold = RNG.gamma(
        shape=8, scale=0.15
    )
    t0 = truncated_normal_rvs(0.3, 0.1, lower=0, random_state=RNG)

    return {"v_intercept": drift_intercept, "v_slope": drift_slope, "s_true": sd_true, "b": threshold, "t0": t0}


def rdm_experiment_simple(
    v_intercept,
    v_slope,
    s_true,
    b,
    t0,
    num_obs
):
    """Simulates data from a single subject in a multi-alternative response times experiment."""
    num_accumulators = 2

    # Acc1 = false, Acc2 = true
    v = np.hstack([v_intercept, v_intercept + v_slope])
    s = np.hstack([1.0, s_true])

    mu = b / v
    lam = (b / s) ** 2

    # First passage time
    fpt = np.zeros((num_accumulators, num_obs))
    
    for i in range(num_accumulators):
        fpt[i, :] = RNG.wald(mu[i], lam[i], size=num_obs)

    resp = fpt.argmin(axis=0)
    rt = fpt.min(axis=0) + t0

    return {"x": np.c_[rt, resp]}


def meta(batch_size):
    return dict(num_obs=100) # Number of trials per dataset

# Number of simulated datasets; should be large; results in datasets with batch_size x num_obs rows
BATCH_SIZE_TRAINING = 5000 

simulator = bf.make_simulator([prior, rdm_experiment_simple], meta_fn=meta)

# Create training set
data = simulator.sample(BATCH_SIZE_TRAINING)

df = (pl.DataFrame(data)
    .explode("x")
    .with_columns(
        pl.col("x").arr.get(0).alias("rt"),
        pl.col("x").arr.get(1).alias("resp")
    )
    .drop(["x", "num_obs"])
    .explode(["v_intercept", "v_slope", "s_true", "b", "t0"])
)

df.write_csv(os.path.join("data", f"rdm_data_training.csv"))

# Create validation set
data = simulator.sample(int(0.2 * BATCH_SIZE_TRAINING))

df = (pl.DataFrame(data)
    .explode("x")
    .with_columns(
        pl.col("x").arr.get(0).alias("rt"),
        pl.col("x").arr.get(1).alias("resp")
    )
    .drop(["x", "num_obs"])
    .explode(["v_intercept", "v_slope", "s_true", "b", "t0"])
)

df.write_csv(os.path.join("data", f"rdm_data_validation.csv"))
