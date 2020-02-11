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
  object$input$value <- newdata
  cg_graph_forward(object$graph, object$output)
  object$output$value
}
