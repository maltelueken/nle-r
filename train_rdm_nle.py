import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
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
    return dict(num_obs=100)


simulator = bf.make_simulator([prior, rdm_experiment_simple], meta_fn=meta)

par_names = ["v_intercept", "v_slope", "s_true", "b", "t0"]

adapter = (
    bf.Adapter()
    .to_array()
    .convert_dtype("float64", "float32")
    .expand_dims(par_names, axis=1)
    .broadcast(par_names, to="x")
    .concatenate(par_names, into="inference_conditions")
    .rename("x", "inference_variables")
    .keep(
        ["inference_variables", "inference_conditions"]
    )
)

inference_network = bf.networks.CouplingFlow()

checkpoint_path = "checkpoints/model.keras"

if (os.path.exists(checkpoint_path)):
    approximator = keras.saving.load_model(checkpoint_path)
else:
    approximator = bf.ContinuousApproximator(
        inference_network=inference_network,
        adapter=adapter
    )

epochs = 1
num_batches = 500
batch_size = 64
learning_rate = keras.optimizers.schedules.CosineDecay(5e-4, decay_steps=epochs*num_batches, alpha=1e-6)
optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
approximator.compile(optimizer=optimizer)

history = approximator.fit(
    epochs=epochs,
    num_batches=num_batches,
    batch_size=batch_size,
    simulator=simulator,
    callbacks=[keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        monitor="loss",
        mode="min",
        save_best_only=True
    )]
)
