
invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

Sys.setenv(RICU_CONFIG_PATH = file.path(here::here(), "config", "ricu-dict"))

sepsis3_crit <- function(source, pids = NULL, keep_components = TRUE,
         dat = NULL) {
  
  if (is.null(dat)) {
    dat <- load_concepts("sofa", source, patient_ids = pids,
                         keep_components = keep_components)
  } else if (!is_ts_tbl(dat)) {
    dat <- data.table::copy(dat[["sofa"]])
  }
  
  stopifnot(is_ts_tbl(dat))
  
  if (grepl("eicu", source)) {
    
    si <- load_concepts("susp_inf", source, abx_min_count = 2L,
                        positive_cultures = TRUE, id_type = "icustay",
                        patient_ids = pids, si_mode = "or",
                        keep_components = keep_components)
    
  } else if (identical(source, "hirid")) {
    
    si <- load_concepts("susp_inf", source, abx_min_count = 2L,
                        id_type = "icustay", patient_ids = pids,
                        si_mode = "or", keep_components = keep_components)
    
  } else {
    
    si <- load_concepts("susp_inf", source, id_type = "icustay",
                        patient_ids = pids, keep_components = keep_components)
  }
  
  sep3(dat, si, si_window = "any", keep_components = keep_components,
       source = source)
}

sep3 <- function (..., si_window = c("first", "last", "any"), delta_fun = delta_cummin, 
                  sofa_thresh = 2L, si_lwr = hours(48L), si_upr = hours(24L), 
                  keep_components = TRUE, interval = NULL, source = NULL) 
{
  cnc <- c("sofa", "susp_inf")
  res <- ricu:::collect_dots(cnc, interval, ...)
  assert_that(is.count(sofa_thresh), is.flag(keep_components), 
              ricu:::not_null(delta_fun))
  si_lwr <- ricu:::as_interval(si_lwr)
  si_upr <- ricu:::as_interval(si_upr)
  delta_fun <- ricu:::str_to_fun(delta_fun)
  si_window <- match.arg(si_window)
  sofa <- res[["sofa"]]
  susp <- res[["susp_inf"]]
  id <- id_vars(sofa)
  ind <- index_var(sofa)
  sus_cols <- setdiff(data_vars(susp), "susp_inf")
  sofa <- sofa[, `:=`(c("join_time1", "join_time2"), list(get(ind), 
                                                          get(ind)))]
  on.exit(rm_cols(sofa, c("join_time1", "join_time2"), by_ref = TRUE))
  susp <- susp[is_true(get("susp_inf")), ]
  susp <- susp[, `:=`(c("susp_inf"), NULL)]
  susp <- susp[, `:=`(c("si_lwr", "si_upr"), list(get(index_var(susp)) - 
                                                    si_lwr, get(index_var(susp)) + si_upr))]
  if (si_window %in% c("first", "last")) {
    susp <- dt_gforce(susp, si_window, id)
  }
  join_clause <- c(id, "join_time1 >= si_lwr", "join_time2 <= si_upr")
  res <- sofa[susp, c(list(delta_sofa = delta_fun(get("sofa"))), 
                      mget(c(ind, sus_cols))), on = join_clause, by = .EACHI, 
              nomatch = NULL]
  res <- res[is_true(get("delta_sofa") >= sofa_thresh), ]
  cols_rm <- c("join_time1", "join_time2")
  if (!keep_components) {
    cols_rm <- c(cols_rm, "delta_sofa")
  }
  res <- rm_cols(res, cols_rm, by_ref = TRUE)
  res <- res[, head(.SD, n = 1L), by = c(id_vars(res))]
  res <- res[, `:=`(c("sep3"), TRUE)]
  
  if (!(grepl("eicu", source) | source == "hirid")) {
    
    res[, cs3t := pmax(get(index_var(res)), abx_time, samp_time)]
    
  } else {
    
    # solve the sampling case
    res_samp <- res[!is.na(samp_time)]
    res_samp[, cs3t := pmax(get(index_var(res_samp)), abx_time, samp_time, 
                            na.rm = TRUE)]
    res_samp <- res_samp[, c(id_vars(res_samp), "cs3t"), with=FALSE]
    
    # go to abx case
    res_abx <- res[is.na(samp_time)]
    res_abx <- res_abx[, c(id_var(res_abx), "abx_time"), with=F]
    add_abx <- load_concepts("abx", source, aggregate = "sum")
    
    res_abx <- merge(add_abx, res_abx)
    res_abx <- res_abx[(get(index_var(res_abx)) > abx_time) |
                         (get(index_var(res_abx)) >= abx_time & abx > 1),
                       head(.SD, n = 1L),
                       by = c(id_vars(res_abx))]
    
    res_abx <- rename_cols(res_abx, "later_abx", index_var(res_abx))
    res_abx <- as_id_tbl(res_abx)
    res_abx <- res_abx[, c(id_vars(res_abx), "later_abx"), with = FALSE]
    
    res_abx <- merge(res[is.na(samp_time)], res_abx)
    res_abx[, cs3t := pmax(get(index_var(res_abx)), abx_time, later_abx)]
    res_abx <- res_abx[, c(id_vars(res_abx), "cs3t"), with=FALSE]
    
    res <- rbind(res_samp, res_abx) 
    
  }
  
  res <- res[, c(id_vars(res), "cs3t"), with=FALSE]
  res <- rename_cols(res, "stay_id", id_var(res))
  res
}

for (src in c("mimic", "eicu", "hirid", "aumc")) {
  
  cs3t <- sepsis3_crit(src)
  arrow::write_parquet(
    cs3t, 
    sink = here::here("results", paste0(src, "_cs3t.parquet"))
  )
  
}
