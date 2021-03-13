
read_to_bm <- function(...) read_to_mat(..., mat_type = "big")

read_to_mat <- function(source, path = data_path("mm"), cols = feature_set(),
                        norm_cols = norm_sel(cols), split = "split_0",
                        pids = coh_split(source, split = split),
                        mat_type = c("mem", "big")) {

  if (length(norm_cols) > 0L) {
    sta <- read_colstats(source, split, norm_cols)
  } else {
    sta <- NULL
  }

  file <- arrow::open_dataset(file.path(path, source))
  subs <- dplyr::filter(file, stay_id %in% pids)$filtered_rows

  scan <- file$NewScan()$Filter(subs)$Project(cols)$Finish()
  tble <- scan$ToTable()

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

    if (col %in% rownames(sta)) {
      tmp <- zero_impute(
        zscore(tmp, sta[col, "means"], sta[col, "stds"])
      )
    }

    res[, col] <- tmp
  }

  res
}

read_to_df <- function(source, path = data_path("mm"), cols = feature_set(),
                       norm_cols = norm_sel(cols), split = "split_0",
                       pids = coh_split(source, split = split)) {

  file <- arrow::open_dataset(file.path(path, source))

  if (length(pids) > 0L && length(cols) > 0L) {
    res <- dplyr::filter(file, stay_id %in% pids)
    res <- dplyr::select(res, dplyr::all_of(cols))
  } else if (length(pids) > 0L) {
    res <- dplyr::filter(file, stay_id %in% pids)
  } else if (length(cols) > 0L) {
    res <- dplyr::select(file, dplyr::all_of(cols))
  } else {
    res <- file
  }

  res <- dplyr::collect(res)

  if (length(norm_cols) > 0L) {
    res <- data.table::setDT(res)
    sta <- read_colstats(source, split, norm_cols)
    fea <- rownames(sta)
    res <- res[, c(fea) := Map(zscore, .SD, sta[, "means"], sta[, "stds"]),
               .SDcols = fea]
    res <- res[, c(fea) := lapply(.SD, zero_impute), .SDcols = fea]
    res <- data.table::setDF(res)
  }

  try_id_tbl(res)
}

read_to_vec <- function(source, path = data_path("mm"), col = "sep3",
                        pids = coh_split(source)) {

  read_to_df(source, path, col, pids)[[col]]
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

create_parquet <- function(x, name, ...) {
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
    paste0(source, "_[0-9]\\.[0-9]\\.[0-9]\\.parquet"),
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

try_id_tbl <- function(x) {

  if ("stay_id" %in% colnames(x)) {

    if ("stay_time" %in% colnames(x)) {

      x <- data.table::set(x, j = "stay_time",
        value = as.difftime(x[["stay_time"]], units = "hours")
      )

      x <- as_ts_tbl(x, id_vars = "stay_id", index_var = "stay_time",
                     interval = hours(1L), by_ref = TRUE)

    } else {

      x <- as_id_tbl(x, id_vars = "stay_id", by_ref = TRUE)
    }
  }

  x
}
