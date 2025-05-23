import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
import polars as pl

import bayesflow as bf


PAR_NAMES = ["v_intercept", "v_slope", "s_true", "b", "t0"]

NUM_TRIALS = 100


def load_data(filename):
    df = pl.read_csv(filename)

    data = {}

    for key in PAR_NAMES:
        data[key] = np.reshape(np.array(df[key]), (-1, NUM_TRIALS)).mean(axis=1, keepdims=True)

    data["x"] = np.reshape(np.vstack([df["rt"], df["resp"]]).T, (-1, NUM_TRIALS, 2))

    return data


adapter = (
    bf.Adapter()
    .to_array()
    .convert_dtype("float64", "float32")
    .expand_dims(PAR_NAMES, axis=1)
    .broadcast(PAR_NAMES, to="x")
    .concatenate(PAR_NAMES, into="inference_conditions")
    .rename("x", "inference_variables")
    .keep(
        ["inference_variables", "inference_conditions"]
    )
)

data_training = load_data(os.path.join("data", f"rdm_data_training.csv"))
data_validation = load_data(os.path.join("data", f"rdm_data_validation.csv"))

dataset_training = bf.OfflineDataset(data=data_training, batch_size=100, adapter=adapter)
dataset_validation = bf.OfflineDataset(data=data_validation, batch_size=100, adapter=adapter)

inference_network = bf.networks.CouplingFlow()

checkpoint_path = "checkpoints/model_offline.keras"

if (os.path.exists(checkpoint_path)):
    approximator = keras.saving.load_model(checkpoint_path)
else:
    approximator = bf.ContinuousApproximator(
        inference_network=inference_network,
        adapter=adapter
    )

epochs = 5
num_batches = dataset_training.num_batches
learning_rate = keras.optimizers.schedules.CosineDecay(5e-4, decay_steps=epochs*num_batches, alpha=1e-6)
optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
approximator.compile(optimizer=optimizer)

history = approximator.fit(
    dataset=dataset_training,
    epochs=epochs,
    callbacks=[keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        monitor="loss",
        mode="min",
        save_best_only=True
    )],
    validation_dataset=dataset_validation
)
