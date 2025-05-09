# This is an example of how to load a NLE trained with BayesFlow for inference into R

library(keras3)

# Create new virual env
reticulate::virtualenv_create("bayesflow") # Only first time
reticulate::use_virtualenv("bayesflow")

# Install BayesFlow and Tensorflow (only first time)
reticulate::py_install("bayesflow==2.0.0") # Latest version didn't work
reticulate::py_install("tensorflow")

# BF and TF must be imported before loading otherwise R crashes fatally
tf <- reticulate::import("tensorflow")
bf <- reticulate::import("bayesflow")

# See if GPU is available
tf$test$is_gpu_available()

# Load NLE approximator
approximator <- load_model("checkpoints/model.keras")

# Create some fake data
data <- list(
  x = np_array(array(matrix(rgamma(200, shape = 2), 100, 2), dim = c(10, 10, 2))),
  v_intercept = np_array(matrix(rgamma(10, shape = 2), 10, 1)),
  v_slope = np_array(matrix(rgamma(10, shape = 2), 10, 1)),
  s_true = np_array(matrix(rgamma(10, shape = 2), 10, 1)),
  b = np_array(matrix(rgamma(10, shape = 2), 10, 1)),
  t0 = np_array(matrix(rgamma(10, shape = 2), 10, 1))
)

# Calculate LL of data
log_lik <- approximator$log_prob(data)

# Sum of LL per dataset
apply(log_lik, 2, sum)
