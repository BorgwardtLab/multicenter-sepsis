
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

read_var_json <- function(path = cfg_path("variables.json")) {

  get_one <- function(x, i) {
    if (is.null(x[[i]][[1L]])) NA_character_ else x[[i]][[1L]]
  }

  get_chr <- function(x, i) vapply(x, get_one, character(1L), i)

  cfg <- jsonlite::read_json(path)
  col <- mget(paste0("col_", get_chr(cfg, 3L)), asNamespace("readr"))

  data.frame(
    concept = get_chr(cfg, 1L),
    callenge = get_chr(cfg, 2L),
    col_spec = I(lapply(col, do.call, list()))
  )
}
