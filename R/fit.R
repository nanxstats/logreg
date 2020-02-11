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
  graph <- cg_graph(eager = FALSE)

  # initialize input (X), target (y)
  input <- cg_constant(x, "input")
  target <- cg_constant(y, "target")

  # intialize parameters (beta, alpha)
  parms <- list(
    beta = cg_parameter(initialize_glorot_normal(ncol(x), 1), "beta"),
    alpha = cg_parameter(initialize_constant(0), "alpha")
  )

  # define the model
  output <- cg_sigmoid(cg_linear(input, parms$beta, parms$alpha), "output")

  # define cost function
  loss <- cg_mean(crossentropy(target, output), "loss")

  # track errors
  error <- rep(0, n_epochs)

  # optimize parameters via sgd
  for (i in 1:n_epochs) {
    cg_graph_forward(graph, loss)
    cg_graph_backward(graph, loss)
    for (parm in parms) parm$value <- parm$value - learning_rate * parm$grad
    error[i] <- loss$value
  }

  lst <- list("graph" = graph, "input" = input, "output" = output, "error" = error)
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
  graph <- cg_graph(eager = FALSE)

  # initialize input (X), target (y)
  input <- cg_constant(x, "input")
  target <- cg_constant(y, "target")

  # intialize parameters (beta, alpha)
  parms <- list(
    beta = cg_parameter(initialize_glorot_normal(ncol(x), 1), "beta"),
    alpha = cg_parameter(initialize_constant(0), "alpha")
  )

  # define the model
  output <- cg_sigmoid(cg_linear(input, parms$beta, parms$alpha), "output")

  # define cost function
  loss <- cg_add(
    cg_mean(crossentropy(target, output)),
    cg_mean(lambda * cg_sum(parms$beta^2)) / 2,
    "loss"
  )

  # track errors
  error <- rep(0, n_epochs)

  # optimize parameters via sgd
  for (i in 1:n_epochs) {
    cg_graph_forward(graph, loss)
    cg_graph_backward(graph, loss)
    for (parm in parms) parm$value <- parm$value - learning_rate * parm$grad
    error[i] <- loss$value
  }

  lst <- list("graph" = graph, "input" = input, "output" = output, "error" = error)
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
  graph <- cg_graph(eager = FALSE)

  # initialize input (X), target (y)
  input <- cg_constant(x, "input")
  target <- cg_constant(y, "target")

  # intialize parameters (beta, alpha)
  parms <- list(
    beta = cg_parameter(initialize_glorot_normal(ncol(x), 1), "beta"),
    alpha = cg_parameter(initialize_constant(0), "alpha")
  )

  # define the model
  output <- cg_sigmoid(cg_linear(input, parms$beta, parms$alpha), "output")

  # define the SELO (seamless l0) loss
  loss <- cg_add(
    cg_mean(crossentropy(target, output)),
    cg_mean(cg_ln((cg_abs(parms$beta) / (cg_abs(parms$beta) + tau)) + 1)) / log(2),
    "loss"
  )

  # track errors
  error <- rep(0, n_epochs)

  # optimize parameters via sgd
  for (i in 1:n_epochs) {
    cg_graph_forward(graph, loss)
    cg_graph_backward(graph, loss)
    for (parm in parms) parm$value <- parm$value - learning_rate * parm$grad
    error[i] <- loss$value
  }

  lst <- list("graph" = graph, "input" = input, "output" = output, "error" = error)
  class(lst) <- "logreg"
  lst
}
