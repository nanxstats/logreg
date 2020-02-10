#' plot errors
#'
#' @param x model object
#'
#' @export plot_error
#'
#' @importFrom graphics plot abline
plot_error <- function(x) plot(x$error, type = "l", xlab = "Epoch", ylab = "Error")

#' plot estimated coefficients
#'
#' @param x model object
#'
#' @export plot_coef
plot_coef <- function(x) {
  coef <- if ("msaenet" %in% class(x)) x$beta else cg_graph_get(x$graph, "beta")$value
  plot(1:length(coef), coef, type = "h", xlab = "Variable Index", ylab = "Coefficient")
  abline(h = 0)
}
