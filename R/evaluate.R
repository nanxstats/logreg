#' compute auc
#'
#' @param label true label
#' @param prob predicted probability
#'
#' @export auc
#'
#' @importFrom pROC roc
auc <- function(label, prob) as.numeric(pROC::roc(as.numeric(label), as.numeric(prob), quiet = TRUE)$auc)
