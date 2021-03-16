
read_to_bm <- function(...) read_train(..., mat_type = "big")

read_to_df <- function(...) read_train(..., mat_type = "df")

read_to_mat <- function(...) read_train(..., mat_type = "mem")

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
  subs <- dplyr::filter(file, stay_id %in% pids)$filtered_rows

  scan <- file$NewScan()$Filter(subs)$Project(cols)$Finish()
  tble <- scan$ToTable()

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

read_res <- function(train_src = "mimic_demo", test_src = train_src,
                     feat_set = c("basic", "wav", "sig", "full"),
                     predictor = c("linear", "rf"),
                     target = c("class", "hybrid", "reg"),
                     dir = data_path("res"), jobid = NULL) {

  if (is.null(jobid)) {
    dir <- grep(
      paste0("^", file.path(dir, "model_"), "[0-9]+"), list.dirs(dir),
      value =TRUE
    )[1L]
  } else {
    dir <- file.path(dir, paste0("model_", jobid))
  }

  fil <- list.files(dir,
    paste(predictor, target, feat_set, train_src, test_src, sep = "-"),
    full.names = TRUE
  )

  if (length(fil) == 0L) return(NULL)

  res <- jsonlite::read_json(fil, simplifyVector = TRUE, flatten = TRUE)

  res <- data.frame(
    stay_id = rep(as.integer(names(res$times)), lengths(res$times)),
    stay_time = do.call(c, res$times),
    prediction = do.call(c, res$scores),
    label = do.call(c, res$labels)
  )

  try_id_tbl(res)
}
