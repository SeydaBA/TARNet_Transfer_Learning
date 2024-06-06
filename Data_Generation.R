
library(mvtnorm)
library(readr)
args <- commandArgs(trailingOnly = TRUE)
sim_id <- as.numeric(args[1])
print(paste("sim_id:", sim_id)) 
set.seed(42)

generate_source_data <- function(N.s, p.s) {
  phi.s <- diag(p.s) * .5 + .5 
  betaX.s <- rep(.3, p.s) 
  betaXA.s <- rep(.05, p.s) 
  betaA.s <- 1 
  
  X.t <- rmvnorm(N.s, sigma = phi.s) 
  A.t <- sample(c(0, 1), N.s, replace = TRUE)

  Y0.t <- X.t %*% betaX.s + rnorm(N.s, 0, 1)
  Y1.t <- Y0.t + betaA.s + X.t %*% betaXA.s

  Y.t <- Y0.t * (1 - A.t) + Y1.t * A.t
  
  data.frame(X.t, A.t, Y0.t, Y1.t, Y.t)
}


generate_random_target_data <- function(source_data, N.t, seed) {
  set.seed(seed)
  N.s <- nrow(source_data)
  samp <- sample(N.s, N.t, replace = TRUE)
  source_data[samp, ]
  colnames(target_data) <- colnames(source_data)
  return(target_data)
}


generate_biased_target_data <- function(source_data, N.t, seed) {
  set.seed(seed)
  X.s <- source_data[, 1:5]
  A.s <- as.numeric(source_data$A.t)
  Y0.s <- as.numeric(source_data$Y0.t)
  N.s <- nrow(source_data)
  
  expit <- function(x) (1 / (1 + exp(-x)))
  p.select <- expit(Y0.s)
  cp.select <- p.select * A.s + (1 - p.select) * (1 - A.s)
  
  N.tt <- 1
  dat.t <- matrix(NA, N.t, ncol(source_data))
  while (N.tt <= N.t) {
    sampi <- sample(N.s, 1)
    sampii <- sample(c(0, 1), 1, prob = c(1 - cp.select[sampi], cp.select[sampi]))
    if (sampii == 1) {
      dat.t[N.tt, ] <- as.numeric(source_data[sampi, ])
      N.tt <- N.tt + 1
    }
  }
  
  dat.t <- as.data.frame(dat.t)
  colnames(dat.t) <- colnames(source_data)
  return(dat.t)
}


source_1k <- generate_source_data(1000, 5)
source_5k <- generate_source_data(5000, 5)


observation_counts <- c(50, 100, 250)
#types <- c("biased")
types <- c("biased")

num_combinations <- length(observation_counts) * length(types) * 100
print(paste("num_combinations:", num_combinations)) 

if (!is.na(sim_id) && sim_id <= num_combinations) {
  source_data <- source_5k
  source_name <- "5k"
  local_sim_id <- sim_id
} else if (!is.na(sim_id)) {
  source_data <- source_1k
  source_name <- "1k" 
  local_sim_id <- sim_id - num_combinations
} else {
  stop("Invalid sim_id or num_combinations")
}


obs_index <- ((local_sim_id - 1) %/% 100) %% length(observation_counts) + 1
type_index <- ((local_sim_id - 1) %/% (100 * length(observation_counts))) + 1
version <- (local_sim_id - 1) %% 100 + 1

obs_count <- observation_counts[obs_index]
type <- types[type_index]
seed <- 42 + version


if (type == "biased") {
  target_data <- generate_random_target_data(source_data, obs_count, seed)
} else {
  target_data <- generate_biased_target_data(source_data, obs_count, seed)
}


file_name <- sprintf("dataset_%s_target_%s_%d_%d.csv", source_name, type, obs_count, version)
write.csv(target_data, file_name, row.names = FALSE)

