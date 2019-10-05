#' make predictions from a fitted model
#'
#' @param object model object
#' @param newdata new predictor matrix
#' @param ... unused
#'
#' @method predict logreg
#'
#' @export
predict.logreg <- function(object, newdata, ...) {
  pred <- cg_graph_run(object$graph, object$loss, list("input" = newdata, "target" = 0))
  pred$output
}
