#' contant initializer
#'
#' @param value contant value
#'
#' @export initialize_constant
initialize_constant <- function(value = 0) value

#' Glorot/Xavier normal initializer
#'
#' @param fan_in number of input elements
#' @param fan_out number of output elements
#' @param seed random number seed
#'
#' @export initialize_glorot_normal
#'
#' @importFrom stats rnorm
initialize_glorot_normal <- function(fan_in, fan_out, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  stddev <- sqrt(2 / (fan_in + fan_out))
  matrix(rnorm(fan_in * fan_out, mean = 0, sd = stddev), fan_in, fan_out)
}
