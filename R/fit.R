#' fit a logistic regression model
#'
#' @param x predictor matrix (n x p)
#' @param y response matrix (n x 1)
#' @param learning_rate learning rate
#' @param n_epochs number of epoches
#'
#' @export fit_logistic
#'
#' @import cgraph

fit_logistic <- function(x, y, learning_rate = 0.05, n_epochs = 1) {
  y <- as.numeric(y)

  # y = beta X + alpha
  graph <- cg_graph()

  # initialize input (X), target (y)
  input <- cg_input("input")
  target <- cg_input("target")

  # intialize parameters (beta, alpha)
  parms <- list(
    beta = cg_parameter(initialize_glorot_normal(ncol(x), 1), "beta"),
    alpha = cg_parameter(initialize_constant(0), "alpha")
  )

  # define the model
  output <- cg_sigmoid(cg_add(cg_matmul(input, parms$beta), cg_as_numeric(parms$alpha)), "output")

  # define cost function
  loss <- cg_mean(crossentropy(target, output), "loss")

  # track errors
  error <- rep(0, n_epochs)

  # optimize parameters via sgd
  for (i in 1:n_epochs) {
    values <- cg_graph_run(graph, loss, list(input = x, target = y))
    grads <- cg_graph_gradients(graph, loss, values)
    for (parm in parms) parm$value <- parm$value - learning_rate * grads[[parm$name]]
    error[i] <- values$loss
  }

  lst <- list("graph" = graph, "loss" = loss, "error" = error)
  class(lst) <- "logreg"
  lst
}

#' fit a regularized logistic regression model (ridge penalty)
#'
#' @param x predictor matrix (n x p)
#' @param y response matrix (n x 1)
#' @param learning_rate learning rate
#' @param n_epochs number of epoches
#' @param lambda regularization parameter
#'
#' @export fit_logistic_ridge

fit_logistic_ridge <- function(x, y, learning_rate = 0.05, n_epochs = 1, lambda = 1) {
  y <- as.numeric(y)

  # y = beta X + alpha
  graph <- cg_graph()

  # initialize input (X), target (y)
  input <- cg_input("input")
  target <- cg_input("target")

  # intialize parameters (beta, alpha)
  parms <- list(
    beta = cg_parameter(initialize_glorot_normal(ncol(x), 1), "beta"),
    alpha = cg_parameter(initialize_constant(0), "alpha")
  )

  # define the model
  output <- cg_sigmoid(cg_add(cg_matmul(input, parms$beta), cg_as_numeric(parms$alpha)), "output")

  # define cost function
  loss <- cg_add(
    cg_mean(crossentropy(target, output)),
    cg_div(cg_mean(cg_mul(lambda, cg_sum(cg_pow(parms$beta, 2)))), 2),
    "loss"
  )

  # track errors
  error <- rep(0, n_epochs)

  # optimize parameters via sgd
  for (i in 1:n_epochs) {
    values <- cg_graph_run(graph, loss, list(input = x, target = y))
    grads <- cg_graph_gradients(graph, loss, values)
    for (parm in parms) parm$value <- parm$value - learning_rate * grads[[parm$name]]
    error[i] <- values$loss
  }

  lst <- list("graph" = graph, "loss" = loss, "error" = error)
  class(lst) <- "logreg"
  lst
}

#' fit a regularized logistic regression model (SELO penalty)
#'
#' @param x predictor matrix (n x p)
#' @param y response matrix (n x 1)
#' @param learning_rate learning rate
#' @param n_epochs number of epoches
#' @param tau regularization parameter
#'
#' @export fit_logistic_selo

fit_logistic_selo <- function(x, y, learning_rate = 0.05, n_epochs = 1, tau = 0.1) {
  y <- as.numeric(y)

  # y = beta X + alpha
  graph <- cg_graph()

  # initialize input (X), target (y)
  input <- cg_input("input")
  target <- cg_input("target")

  # intialize parameters (beta, alpha)
  parms <- list(
    beta = cg_parameter(initialize_glorot_normal(ncol(x), 1), "beta"),
    alpha = cg_parameter(initialize_constant(0), "alpha")
  )

  # define the model
  output <- cg_sigmoid(cg_add(cg_matmul(input, parms$beta), cg_as_numeric(parms$alpha)), "output")

  # define the SELO (seamless l0) loss
  loss <- cg_add(
    cg_mean(crossentropy(target, output)),
    cg_div(cg_mean(cg_ln(cg_add(cg_div(cg_abs(parms$beta), cg_add(cg_abs(parms$beta), tau)), 1))), log(2)),
    "loss"
  )

  # track errors
  error <- rep(0, n_epochs)

  # optimize parameters via sgd
  for (i in 1:n_epochs) {
    values <- cg_graph_run(graph, loss, list(input = x, target = y))
    grads <- cg_graph_gradients(graph, loss, values)
    for (parm in parms) parm$value <- parm$value - learning_rate * grads[[parm$name]]
    error[i] <- values$loss
  }

  lst <- list("graph" = graph, "loss" = loss, "error" = error)
  class(lst) <- "logreg"
  lst
}
