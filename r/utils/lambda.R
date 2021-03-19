
f_pos <- function(delta_t) {
  
  assertthat::assert_that(length(delta_t) == 1L)
  if (delta_t < -12) return(-0.05)
  if (delta_t <= -6) return((delta_t + 12) / 6)
  if (delta_t <= 3) return(1 - ((delta_t + 6) / 9))
  
  0
  
}

f_neg <- function(delta_t) {
  
  assertthat::assert_that(length(delta_t) == 1L)
  if (delta_t <= -6) return(0)
  if (delta_t <= 3) return(-2 * (delta_t + 6) / 9)
  
  0
  
}

# range <- -15:5
# plot(
#   range, sapply(range, f_pos), pch = 19, col = "red", lty = 2,
#   ylim = c(-2, 1), main = "Physionet metric"
# )
# 
# lines(
#   range, sapply(range, f_pos), pch = 19, col = "red", lty = 2,
#   ylim = c(-2, 1)
# )
# 
# points(
#   range, sapply(range, f_neg), pch = 19, col = "blue", lty = 2
# )
# lines(
#   range, sapply(range, f_neg), pch = 19, col = "blue", lty = 2
# )


x <- read_parquet("aumc", cols = c("stay_time", "stay_id", "onset"))

lambda_module <- function(x) {
  
  x[, is_case := any(is_true(onset)), by = "stay_id"]
  
  x[is_case == T, onset_time := stay_time[which(!is.na(onset))], by = "stay_id"]
  
  x[, pos := 0]
  x[, neg := 0]
  
  x[is_case == F, "pos"] <- -0.05
  
  x[is_case == T, pos := f_pos(stay_time - onset_time), 
    by = c("stay_id", "stay_time")]
  
  x[is_case == T, neg := f_neg(stay_time - onset_time), 
    by = c("stay_id", "stay_time")]
  
  x[, opt := max(pos, neg), by = c("stay_id", "stay_time")]
  
  denom <- sum(x[is_case == T][["pos"]]) - 2 * sum(x[is_case == T][["neg"]]) +
    sum(x[is_case == T][["opt"]])
  
  numer <- sum(2 * x[is_case == F][["neg"]]) - sum(x[is_case == F][["pos"]]) -
    sum(x[is_case == F][["opt"]])
  
  lambda <- numer / denom
  
  x[is_case == T, pos := lambda * pos]
  x[is_case == T, neg := lambda * neg]
  
  list(lambda = lambda, utility = x[["pos"]] - x[["neg"]])
  
}

# res <- lambda_module(x)
# res[["lambda"]]
# res
