
load_physionet <- function(dir = data_path("physionet2019"),
                           cfg = cfg_path("variables.json")) {

  concepts <- read_var_json(cfg)
  data_dir <- file.path(dir, "training_setB")

  assert(dir.exists(data_dir), msg = "Directory {data_dir} not found.")

  feat_map <- concepts[!is.na(concepts[["callenge"]]), ]
  feats <- do.call(readr::cols,
    setNames(feat_map[["col_spec"]], feat_map[["callenge"]])
  )

  dat <- read_psv(data_dir, col_spec = feats, id_var = "stay_id")

  dat <- dat[, Gender := data.table::fifelse(Gender == 0L, "Female", "Male")]
  dat <- dat[, O2Sat := rowMeans(.SD, na.rm=TRUE),
             .SDcols = c("O2Sat", "SaO2")]
  dat <- dat[, stay_time := as.difftime(ICULOS - 1, units = "hours")]
  dat <- as_ts_tbl(dat, "stay_id", "stay_time")

  sep <- dat[
    (SepsisLabel), list(stay_time = min(stay_time) + 6, sep3 = TRUE),
    by = "stay_id"
  ]

  dat <- rm_cols(dat, c("SaO2", "ICULOS", "SepsisLabel"), by_ref = TRUE)

  feats <- setNames(feat_map[["concept"]], feat_map[["callenge"]])
  feats <- feats[data_vars(dat)]

  dat <- rename_cols(dat, feats, names(feats), by_ref = TRUE)

  dat <- merge(dat, sep, all = TRUE)

  msg("--> loading complete")

  dat
}

load_ricu <- function(source, var_cfg = cfg_path("variables.json"),
                      coh_cfg = cfg_path("cohorts.json"),
                      min_stay_length = hours(4L),
                      min_onset = hours(4L), max_onset = days(7L),
                      cut_case = hours(24L), cut_ctrl = max_onset + cut_case) {

  truncate_dat <- function(dat, win, flt) {

    if (length(flt) > 0L) {
      dat <- dat[!get(id_var(dat)) %in% flt, ]
    }

    if (is_ts_tbl(dat)) {
      repl_meta <- c("stay_id", "stay_time")
    } else {
      repl_meta <- "stay_id"
    }

    dat <- rename_cols(dat, repl_meta, meta_vars(dat), by_ref = TRUE)

    if (is_ts_tbl(dat)) {

      dat  <- dat[, c("join_time") := list(get("stay_time"))]

      join <- c(paste("stay_id ==", id_vars(win)), "join_time <= cuttime")
      dat <- dat[win, on = join, nomatch = NULL]
      dat <- rm_cols(dat, "join_time", by_ref = TRUE)
    }

    dat
  }

  feats <- read_var_json(var_cfg)[["concept"]]
  feats <- feats[!is.na(feats)]
  pids  <- unlist(jsonlite::read_json(coh_cfg)[[source]]$cohort)

  win <- stay_windows(source, id_type = "icustay", win_type = "icustay",
                      in_time = "intime", out_time = "outtime")

  dat <- load_concepts(feats, source, merge_data = FALSE,
                       id_type = "icustay", patient_ids = pids)
  sep <- sepsis3_crit(source, pids, dat)

  sep  <- sep[, c("join_time") := list(get(index_var(sep)))]

  join <- c(paste(id_vars(sep), "==", id_vars(win)), "join_time >= intime",
                                                     "join_time <= outtime")
  new <- sep[win, on = join, nomatch = NULL]
  tmp <- nrow(new)

  msg("--> removing {nrow(sep) - tmp} patients due to onsets",
      " outside of icu stay.\n")

  new <- new[get(index_var(new)) >= min_onset &
             get(index_var(new)) <= max_onset, ]

  msg("--> removing {tmp - nrow(new)} patients due to onsets",
      " outside of [{format_unit(min_onset)},",
      " {format_unit(max_onset)}].\n")

  flt <- win[outtime < min_stay_length, ]

  msg("--> removing {nrow(flt)} patients due to stay length <",
      " {format_unit(min_stay_length)}].\n")

  flt <- c(id_col(flt), setdiff(id_col(sep), id_col(new)))

  sep <- rm_cols(new, setdiff(data_vars(new), "sep3"), by_ref = TRUE)
  sep <- rename_cols(sep, "sep3_time", index_var(sep))
  win <- merge(win, rm_cols(as_id_tbl(sep), "sep3", by_ref = TRUE),
               all.x = TRUE)
  win <- win[, cuttime := data.table::fifelse(
    is.na(sep3_time), pmin(outtime, cut_ctrl), sep3_time + cut_case
  )]

  msg("--> removing up to {sum(win$outtime - win$cuttime)} due",
      " to censoring data {format_unit(cut_case)} after onsets",
      " and {format_unit(cut_ctrl)} into stays.\n")

  win <- rm_cols(win, c("intime", "outtime", "sep3_time"), by_ref = TRUE)
  dat <- lapply(dat, truncate_dat, win, flt)

  is_ts <- vapply(dat, is_ts_tbl, logical(1L))
  is_id <- vapply(dat, is_id_tbl, logical(1L)) & ! is_ts

  dat <- dat[c(which(is_ts), which(is_id))]

  while(length(dat) > 1L) {
    dat[[1L]] <- merge(dat[[1L]], dat[[2L]], all = TRUE)
    dat[[2L]] <- NULL
  }

  dat <- merge(dat[[1L]], sep, all = TRUE)

  msg("--> loading complete")

  dat
}

sepsis3_crit <- function(source, pids = NULL,
  dat = load_concepts("sofa", source, patient_ids = pids)) {

  if (!is_ts_tbl(dat)) {
    dat <- data.table::copy(dat[["sofa"]])
  }

  stopifnot(is_ts_tbl(dat))

  if (grepl("eicu", source)) {

    si <- load_concepts("susp_inf", source, abx_min_count = 2L,
                        positive_cultures = TRUE, id_type = "icustay",
                        patient_ids = pids, si_mode = "or")

  } else if (identical(source, "hirid")) {

    si <- load_concepts("susp_inf", source, abx_min_count = 2L,
                        id_type = "icustay", patient_ids = pids,
                        si_mode = "or")

  } else {

    si <- load_concepts("susp_inf", source, id_type = "icustay",
                        patient_ids = pids)
  }

  sep3(dat, si, si_window = "any")
}

augment <- function(x, fun, suffix,
                    cols = grep("_raw$", colnames(x), value = TRUE),
                    by = NULL, win = NULL, tmpdir = tempdir(),
                    names = sub("_raw$", paste0("_", suffix), cols), ...) {

  inf_to_na <- function(x) replace(x, is.infinite(x), NA)

  msg("--> augmentation step {suffix}")

  if (is.numeric(win)) {
    win <- as.difftime(win, units = "hours")
  }

  if (is.null(win) && is.null(by)) {

    res <- x[, lapply(.SD, fun, ...), .SDcols = c(cols)]

  } else if (is.null(win)) {

    res <- x[, lapply(.SD, fun, ...), .SDcols = c(cols), by = c(by)]

  } else if (is.character(fun)) {

    fix_inf <- FALSE

    id_cols <- id_vars(x)
    ind_col <- index_var(x)

    tmp_col <- c("tmp_ind_1", "tmp_ind_2")

    wins <- x[,
      c(mget(id_cols), list(min_time = get(ind_col) - win,
                            max_time = get(ind_col)))
    ]

    x <- x[, c(tmp_col) := list(get(ind_col), get(ind_col))]
    on.exit(rm_cols(x, tmp_col, by_ref = TRUE))

    join <- paste(c(id_cols, tmp_col),
                  c(rep("==", length(id_cols)), "<=", ">="),
                  c(id_cols, "max_time", "min_time"))

    res <- withCallingHandlers({
      switch(fun,
        min  = x[wins, lapply(.SD, min,  ...), .SDcols = cols, on = join,
                 by = .EACHI],
        max  = x[wins, lapply(.SD, max,  ...), .SDcols = cols, on = join,
                 by = .EACHI],
        mean = x[wins, lapply(.SD, mean, ...), .SDcols = cols, on = join,
                 by = .EACHI],
        var  = x[wins, lapply(.SD, var,  ...), .SDcols = cols, on = join,
                 by = .EACHI]
      )
    }, warning = function(w) {

      if (identical(substr(conditionMessage(w), 1, 24),
                    "no non-missing arguments")) {

        fix_inf <<- TRUE
        invokeRestart("muffleWarning")
      }
    })

    res <- rm_cols(res, tmp_col, by_ref = TRUE)

    if (fix_inf) {
      res <- res[, c(cols) := lapply(.SD, inf_to_na), .SDcols = c(cols)]
    }

  } else {

    res <- slide(x, lapply(.SD, fun, ...), win, .SDcols = c(cols))
  }

  assert(identical(nrow(x), nrow(res)))

  res <- rename_cols(res, names, cols, by_ref = TRUE)
  res <- rm_cols(res, setdiff(colnames(res), names), by_ref = TRUE)

  res
}

prof <- function(expr, envir = parent.frame()) {

  mem <- memuse::Sys.procmem()
  tim <- Sys.time()

  res <- eval(expr, envir = envir)

  cur <- memuse::Sys.procmem()
  cil <- cur[["peak"]] - mem[["peak"]]

  msg("    Runtime: {format(Sys.time() - tim, digits = 4)}")
  if (length(cil)) msg("    Memory ceiling increased by: {as.character(cil)}")
  msg("    Current memory usage: {as.character(cur[['size']])}")

  res
}

export_data <- function(src, dest_dir = data_path("export"),
                        var_cfg = cfg_path("variables.json"), ...) {

  aug_fun_win <- function(fun, win, suf)

  assert(is.string(src), dir.exists(dir))

  dat <- prof(
    if (identical(src, "physionet2019")) {
      load_physionet(cfg = var_cfg)
    } else {
      load_ricu(src, var_cfg = var_cfg, ...)
    }
  )

  dat <- rm_na(dat, meta_vars(dat), "any")
  dat <- fill_gaps(dat)

  dat <- dat[, onset := sep3]
  dat <- replace_na(dat, type = "locf", by_ref = TRUE, vars = "sep3",
                    by = id_vars(dat))
  dat <- replace_na(dat, FALSE, by_ref = TRUE, vars = "sep3")
  dat <- dat[, c("sex") := list(sex == "Female")]

  cfg <- read_var_json(var_cfg)
  cfg <- cfg[!is.na(cfg$concept), ]

  dat <- rename_cols(dat, cfg$name, cfg$concept, by_ref = TRUE,
                     skip_absent = TRUE)

  missing <- !cfg$name %in% data_vars(dat)

  if (any(missing)) {
    dat <- dat[, c(cfg$name[missing]) := NA_real_]
  }

  dat <- data.table::setcolorder(dat, c(meta_vars(dat), cfg$name))

  tsn <- cfg$name[!cfg$category %in% c("static", "baseline")]
  tsv <- paste0(tsn, "_raw")
  dat <- rename_cols(dat, tsv, tsn, by_ref = TRUE)

  ind <- prof(
    augment(dat, Negate(is.na), "ind", tsv)
  )

  lof <- prof(
    augment(dat, data.table::nafill, "locf", tsv, by = id_vars(dat),
            type = "locf")
  )

  dat <- dat[, c(index_var(dat)) := as.double(
    get(index_var(dat)), units = "hours"
  )]

  res <- c(dat, ind, lof)
  res <- data.table::setDT(res)

  create_parquet(res,
    file.path(dest_dir, paste(src, packageVersion("ricu"), sep = "_"))
  )
}
