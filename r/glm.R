#!/usr/bin/env Rscript

#BSUB -W 24:00
#BSUB -n 16
#BSUB -R rusage[mem=16000]
#BSUB -J glm-big-16
#BSUB -o data/glm/glm-big-16_%J.out

n_cores <- function() {

  res <- Sys.getenv("LSB_DJOB_NUMPROC", unset = parallel::detectCores() / 2L)

  message("using ", res, " cores")

  as.integer(res)
}

res_name <- function() {
  res <- paste0(Sys.getenv("LSB_JOBNAME"), "_", Sys.getenv("LSB_JOBID"))
  paste0(if (identical(res, "_")) "results" else res, ".qs")
}

read_to_bm <- function(path, ...) {

  file <- arrow::ParquetFileReader$create(path)

  cols <- names(file$GetSchema())
  sel  <- !cols %in% c("time", "id")

  res <- bigmemory::big.matrix(file$num_rows, sum(sel), type = "double",
                               dimnames = list(NULL, cols[sel]), shared = TRUE)

  cols <- which(sel)

  for (i in seq_along(cols)) {
    res[, i] <- as.double(file$ReadColumn(i)$as_vector())
  }

  res
}

read_response <- function(path) arrow::read_parquet(path)$sep3

path <- file.path("data", "glm")

memuse::Sys.procmem()

dat <- read_to_bm(file.path(path, "X_train.parquet"))
res <- read_response(file.path(path, "y_train.parquet"))

memuse::Sys.procmem()

mod <- biglasso::biglasso(dat, res, family = "binomial", alg.logistic = "MM",
                          ncores = n_cores(), verbose = TRUE)

rm(dat)

memuse::Sys.procmem()

dat <- read_to_bm(file.path(path, "X_validation.parquet"))

memuse::Sys.procmem()

pre <- predict(mod, dat, type = "class")

qs::qsave(list(model = mod, predictions = pre), file.path(path, res_name()))
