
read_to_bm <- function(...) read_train(..., mat_type = "big")

read_to_df <- function(...) read_train(..., mat_type = "df")

read_to_mat <- function(...) read_train(..., mat_type = "mem")

read_to_vec <- function(...) read_train(..., mat_type = "mem")[, 1L]

read_train <- function(source, path = data_path("mm"), cols = feature_set(),
                       norm_cols = norm_sel(cols), split = "split_0",
                       pids = coh_split(source, split = split),
                       mat_type = c("mem", "df", "big")) {

  normalize <- function(x, mean, std) zero_impute(zscore(x, mean, std))
  is_female <- function(x) replace(x, is.na(x), 0)

  if (length(norm_cols) > 0L) {
    sta <- read_colstats(source, split, norm_cols)
  } else {
    sta <- NULL
  }

  file <- arrow::open_dataset(file.path(path, source))
  scan <- file$NewScan()

  if (!is.null(pids)) {
    subs <- dplyr::filter(file, stay_id %in% pids)$filtered_rows
    scan$Filter(subs)
  }

  if (!is.null(cols)) {
    scan$Project(cols)
  }

  tble <- scan$Finish()$ToTable()

  if (identical(match.arg(mat_type), "df")) {

    res <- as.data.frame(tble)
    res <- data.table::setDT(res)

    if (length(norm_cols) > 0L) {

      fea <- rownames(sta)
      res <- res[, c(fea) := Map(normalize, .SD, sta[, "means"],
                                 sta[, "stds"]), .SDcols = fea]
    }

    if ("female" %in% colnames(res)) {
      res <- res[, female := is_female(female)]
    }

    return(res)
  }

  if (identical(match.arg(mat_type), "big")) {

    res <- bigmemory::big.matrix(
      nrow(tble), length(cols), type = "double",
      dimnames = list(NULL, cols), shared = TRUE
    )

  } else {

    res <- matrix(nrow = nrow(tble), ncol = length(cols),
                  dimnames = list(NULL, cols))
  }

  for (col in cols) {

    tmp <- tble$GetColumnByName(col)$as_vector()
    tmp <- as.double(tmp)

    if (identical(col, "female")) {
      tmp <- is_female(tmp)
    } else if (col %in% rownames(sta)) {
      tmp <- normalize(tmp, sta[col, "means"], sta[col, "stds"])
    }

    res[, col] <- tmp
  }

  res
}

read_var_json <- function(path = cfg_path("variables.json")) {

  get_one <- function(x, i) {
    if (length(x[[i]])) x[[i]][[1L]] else NA_character_
  }

  get_chr <- function(x, i) vapply(x, get_one, character(1L), i)

  cfg <- jsonlite::read_json(path)
  col <- mget(paste0("col_", get_chr(cfg, 3L)), asNamespace("readr"))

  data.frame(
    concept = get_chr(cfg, 1L),
    callenge = get_chr(cfg, 2L),
    col_spec = I(lapply(col, do.call, list())),
    name = get_chr(cfg, 4L),
    category = get_chr(cfg, 5L),
    stringsAsFactors = FALSE
  )
}

create_parquet <- function(x, name, attr = NULL, ...) {

  if (!is.null(attr)) {
    x <- arrow::Table$create(x)
    x$metadata <- c(x$metadata,
      lapply(attr, jsonlite::toJSON, dataframe = "columns", auto_unbox = TRUE,
             na = "null")
    )
  }

  arrow::write_parquet(x, paste0(name, ".parquet"), ...)
}

read_parquet <- function(source, dir = data_path(), cols = NULL, pids = NULL) {

  read_subset <- function(n, x, i, j = NULL) {

    res <- x$ReadRowGroup(n)
    res <- res$Filter(res$GetColumnByName("stay_id")$as_vector() %in% i)

    if (!is.null(j)) {
      res <- res$SelectColumns(j)
    }

    as.data.frame(res)
  }

  file <- list.files(
    dir,
    paste0(source, "-[0-9]-[0-9]-[0-9]\\.parquet"),
    full.names = TRUE
  )

  file <- tail(sort(file), n = 1L)

  if (is.null(cols) && is.null(pids)) {

    res <- arrow::read_parquet(file)

  } else {

    readr <- arrow::ParquetFileReader$create(file)

    if (!is.null(cols)) {

      avail <- names(readr$GetSchema())

      assert_that(is.character(cols), all(cols %in% avail))

      cols <- match(cols, avail) - 1L
    }

    if (is.null(pids)) {

      res <- as.data.frame(readr$ReadTable(cols))

    } else {

      res <- lapply(seq.int(readr$num_row_groups) - 1L, read_subset, readr,
                    pids, cols)
      res <- data.table::rbindlist(res)
    }
  }

  try_id_tbl(res)
}

read_meta <- function(file, node = "mcsep") {
  readr <- arrow::ParquetFileReader$create(file)
  readr$GetSchema()$metadata[[node]]
}

y_class <- function(source, left = -6, right = Inf, ...) {

  fpos_module <- function(delta, left, right) {
    is_true(delta >= left & delta <= right)
  }

  dat <- read_to_df(source, cols = c("stay_id", "stay_time", "sep3"),
                    norm_cols = NULL, ...)

  dat <- dat[, onset := stay_time[which(sep3 == 1)[1L]], by = "stay_id"]
  dat <- dat[, delta := stay_time - onset]

  fpos_module(dat[["delta"]], left, right)
}

y_reg <- function(source, split = "split_0", ...) {

  dat <- read_to_df(source,
    cols = c("stay_id", "stay_time", "sep3", "utility"),
    norm_cols = NULL, ..., split = split
  )

  lmb <- read_lambda(source, split)
  dat <- dat[, is_case := any(sep3 == 1L), by = "stay_id"]
  dat <- dat[, lambda := data.table::fifelse(is_case, lmb, 1)]

  dat[["utility"]] * dat[["lambda"]]
}

y_reg2 <- function(source, left = -12, right = 6, mid = -6, u_fp = -0.05,
                   split = "split_0", ...) {

  fpos_module <- function(delta, left, right, mid, u_fp) {

    data.table::fifelse(
      delta < left, u_fp,
      data.table::fifelse(
        delta < mid, (delta - left) / (mid - left),
        data.table::fifelse(delta == mid, 1,
          data.table::fifelse(
            delta <= right, 1 - (delta - mid) / (right - mid), 0
          )
        )
      ), u_fp
    )
  }

  dat <- read_to_df(source,
    cols = c("stay_id", "stay_time", "sep3"),
    norm_cols = NULL, ..., split = split
  )

  dat <- dat[, onset := stay_time[which(sep3 == 1)[1L]], by = "stay_id"]
  dat <- dat[, delta := stay_time - onset]

  fpos_module(dat[["delta"]], left, right, mid, u_fp)
}
