#' binary cross entropy
#'
#' @param y true label (0/1)
#' @param p predicted probability
#'
#' @note \code{-(y * log(p) + (1 - y) * log (1 - p))}
#'
#' @export crossentropy
crossentropy <- function(y, p) {
  -(y * cg_ln(p) + (1 - y) * cg_ln(1 - p))
}
