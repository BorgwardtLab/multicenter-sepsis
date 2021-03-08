
read_to_bm <- function(source, path = data_path("mm"), cols = feature_set(),
                       pids = coh_split(source)) {

  file <- arrow::open_dataset(file.path(path, source, "features"))
  subs <- dplyr::filter(file, stay_id %in% pids)$filtered_rows

  scan <- file$NewScan()$Filter(subs)

  col_dat <- scan$Project(cols[1L])$Finish()$ToTable()

  res <- bigmemory::big.matrix(
    nrow(col_dat), length(cols), type = "double", dimnames = list(NULL, cols),
    shared = TRUE
  )

  res[, cols[1L]] <- as.double(col_dat[[1L]])

  for (col in cols[-1L]) {
    col_dat <- scan$Project(col)$Finish()$ToTable()
    res[, col] <- as.double(col_dat[[1L]])
  }

  res
}

read_to_df <- function(source, path = data_path("mm"), cols = feature_set(),
                       pids = coh_split(source)) {

  file <- arrow::open_dataset(file.path(path, source, "features"))

  res <- dplyr::filter(file, stay_id %in% pids)
  res <- dplyr::select(res, dplyr::all_of(cols))

  dplyr::collect(res)
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

read_parquet <- function(name, cols = NULL, pids = NULL) {

  read_subset <- function(n, x, i, j = NULL) {

    res <- x$ReadRowGroup(n)
    res <- res$Filter(res$GetColumnByName("stay_id")$as_vector() %in% i)

    if (!is.null(j)) {
      res <- res$SelectColumns(j)
    }

    as.data.frame(res)
  }

  file <- list.files(
    dirname(name),
    paste0(basename(name), "_[0-9]\\.[0-9]\\.[0-9]\\.parquet"),
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

  if ("stay_id" %in% colnames(res)) {

    if ("stay_time" %in% colnames(res)) {

      res[["stay_time"]] <- as.difftime(res[["stay_time"]], units = "hours")

      res <- as_ts_tbl(res, id_vars = "stay_id", index_var = "stay_time",
                       interval = hours(1L), by_ref = TRUE)

    } else {

      res <- as_id_tbl(res, id_vars = "stay_id", by_ref = TRUE)
    }
  }

  res
}
