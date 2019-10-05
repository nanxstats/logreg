#' binary cross entropy
#'
#' @param y true label (0/1)
#' @param p predicted probability
#'
#' @note \code{-(y * log(p) + (1 - y) * log (1 - p))}
#'
#' @export crossentropy
crossentropy <- function(y, p) {
  y <- cg_as_numeric(y)
  p <- cg_as_numeric(p)
  cg_neg(cg_add(cg_mul(y, cg_ln(p)), cg_mul(cg_sub(1, y), cg_ln(cg_sub(1, p)))))
}
