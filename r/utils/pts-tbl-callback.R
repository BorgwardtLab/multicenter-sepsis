
DM_callback <- function(x, ...) {
  #browser()
  sub_var <- setdiff(names(x), meta_vars(x))
  if (sub_var == "icd9code") {

    x[, c(sub_var) := gsub(",.*", "", get(sub_var))]

  }

  intm <- data.frame(
    pid = id_col(x),
    icd9 = x[[setdiff(names(x), id_vars(x))]]
  )
  intm <- rowSums(comorbid_charlson(intm)[, c("DM", "DMcx")]) > 0

  res <- id_tbl(
    id = as.integer(names(intm)),
    val = intm, id_vars = "id"
  )

  names(res) <- names(x)

  res
}

CPD_callback <- function(x, ...) {

  sub_var <- setdiff(names(x), meta_vars(x))
  if (sub_var == "icd9code") {

    x[, c(sub_var) := gsub(",.*", "", get(sub_var))]

  }

  intm <- data.frame(
    pid = id_col(x),
    icd9 = x[[setdiff(names(x), id_vars(x))]]
  )
  intm <- rowSums(comorbid_charlson(intm)[, c("Pulmonary"), drop = F]) > 0

  res <- id_tbl(
    id = as.integer(names(intm)),
    val = intm, id_vars = "id"
  )

  names(res) <- names(x)

  res
}

LD_callback <- function(x, ...) {

  sub_var <- setdiff(names(x), meta_vars(x))
  if (sub_var == "icd9code") {

    x[, c(sub_var) := gsub(",.*", "", get(sub_var))]

  }

  intm <- data.frame(
    pid = id_col(x),
    icd9 = x[[setdiff(names(x), id_vars(x))]]
  )
  intm <- rowSums(comorbid_charlson(intm)[, c("LiverMild", "LiverSevere"), drop = F]) > 0

  res <- id_tbl(
    id = as.integer(names(intm)),
    val = intm, id_vars = "id"
  )

  names(res) <- names(x)

  res
}

CRF_callback <- function(x, ...) {

  sub_var <- setdiff(names(x), meta_vars(x))
  if (sub_var == "icd9code") {

    x[, c(sub_var) := gsub(",.*", "", get(sub_var))]

  }

  intm <- data.frame(
    pid = id_col(x),
    icd9 = x[[setdiff(names(x), id_vars(x))]]
  )
  intm <- rowSums(comorbid_charlson(intm)[, c("Renal"), drop = F]) > 0

  res <- id_tbl(
    id = as.integer(names(intm)),
    val = intm, id_vars = "id"
  )

  names(res) <- names(x)

  res
}

Cancer_callback <- function(x, ...) {

  sub_var <- setdiff(names(x), meta_vars(x))
  if (sub_var == "icd9code") {

    x[, c(sub_var) := gsub(",.*", "", get(sub_var))]

  }

  intm <- data.frame(
    pid = id_col(x),
    icd9 = x[[setdiff(names(x), id_vars(x))]]
  )
  intm <- rowSums(comorbid_charlson(intm)[, c("Cancer", "Mets"), drop = F]) > 0

  res <- id_tbl(
    id = as.integer(names(intm)),
    val = intm, id_vars = "id"
  )

  names(res) <- names(x)

  res
}

is_vent_callback <- function(vent_ind, interval, ...) {
  
  vent_ind <- expand(vent_ind)
  vent_ind[, c(index_var(vent_ind)) := NULL]
  vent_ind <- unique(vent_ind)

  vent_ind[, is_vent := vent_ind]
  vent_ind[, vent_ind := NULL]

  vent_ind

}

is_vaso_callback <- function(..., interval) {

  x <- list(...)[["norepi_equiv"]]
  x[, c(index_var(x)) := NULL]

  x[, is_vaso := (max(norepi_equiv) > 0), by = get(id_var(x))]
  x[, norepi_equiv := NULL]

  unique(x)

}

is_abx_callback <- function(..., interval) {

  x <- list(...)[["abx"]]
  x[, c(index_var(x)) := NULL]

  x[, is_abx := (max(abx)), by = get(id_var(x))]
  x[, abx := NULL]

  unique(x)

}

susp_infff <- function (..., abx_count_win = hours(24L), abx_min_count = 1L, 
  positive_cultures = FALSE, si_mode = c("and", "or"), abx_win = hours(24L), 
  samp_win = hours(72L), by_ref = TRUE, interval = NULL) 
{
  si_mode <- match.arg(si_mode)
  assert_that(is.count(abx_min_count), is.flag(positive_cultures), 
    ricu:::is_interval(abx_count_win), 
    ricu:::is_interval(abx_win), ricu:::is_interval(samp_win), 
    is.flag(by_ref))
  cnc <- c("abx", "samp")
  res <- ricu:::collect_dots(cnc, interval, ...)
  if (positive_cultures) {
    samp_fun <- "sum"
  }
  else {
    samp_fun <- quote(list(samp = .N))
  }
  if (!isTRUE(by_ref)) {
    res <- lapply(res, copy)
  }
  switch(si_mode, and = ricu:::si_and, or = ricu:::si_or)(
    ricu:::si_abx(res[["abx"]], 
    abx_count_win, abx_min_count), ricu:::si_samp(aggregate(res[["samp"]], 
      samp_fun)), abx_win, samp_win)
}

susp_inf_mc_callback <- function(...) {

  abx <- list(...)[["abx"]]

  source <- ifelse(
    id_var(abx) == "patientunitstayid",
    "eicu",
    ifelse(
      id_var(abx) == "patientid", "hirid", "other"
    )
  )

  if (grepl("eicu", source)) {

    res <- susp_inf(..., abx_min_count = 2L, positive_cultures = TRUE,
      si_mode = "or")

  } else if (identical(source, "hirid")) {

    res <- susp_inf(..., abx_min_count = 2L, si_mode = "or")

  } else {

    res <- susp_inf(..., abx_min_count = 1L, si_mode = "and")

  }

  res <- rename_cols(res, "susp_inf_mc", "susp_inf")

  res

}

is_si_callback <- function(..., interval) {

  x <- list(...)[["susp_inf_mc"]]
  x[, c(index_var(x)) := NULL]

  x[, is_si := (max(susp_inf_mc)), by = get(id_var(x))]
  x[, susp_inf_mc := NULL]

  unique(x)

}

eth_mim_callback <- function(x, val_var, env) {
  
  groups <- list(
    Caucasian = c("WHITE", "WHITE - BRAZILIAN", "WHITE - EASTERN EUROPEAN", 
                  "WHITE - OTHER EUROPEAN", 
                  "WHITE - RUSSIAN"),
    Asian = c("ASIAN", "ASIAN - ASIAN INDIAN", "ASIAN - CAMBODIAN", "ASIAN - CHINESE", 
              "ASIAN - FILIPINO", "ASIAN - JAPANESE", "ASIAN - KOREAN", "ASIAN - OTHER", 
              "ASIAN - THAI", "ASIAN - VIETNAMESE"),
    Hispanic = c("HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)", 
                 "HISPANIC/LATINO - COLOMBIAN", 
                 "HISPANIC/LATINO - CUBAN", "HISPANIC/LATINO - DOMINICAN", 
                 "HISPANIC/LATINO - GUATEMALAN", 
                 "HISPANIC/LATINO - HONDURAN", "HISPANIC/LATINO - MEXICAN", 
                 "HISPANIC/LATINO - PUERTO RICAN", 
                 "HISPANIC/LATINO - SALVADORAN", "HISPANIC OR LATINO"),
    `African American` = c("BLACK/AFRICAN AMERICAN"),
    Other = c("AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE",
              "UNABLE TO OBTAIN", "UNKNOWN/NOT SPECIFIED", "MIDDLE EASTERN", 
              "MULTI RACE ETHNICITY", 
              "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER", "OTHER", 
              "PATIENT DECLINED TO ANSWER", 
              "PORTUGUESE", "SOUTH AMERICAN",
              "AMERICAN INDIAN/ALASKA NATIVE", 
              "AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE",
              "CARIBBEAN ISLAND", "BLACK/AFRICAN", "BLACK/CAPE VERDEAN", "BLACK/HAITIAN")
  )
  map <- unlist(groups)
  names(map) <- rep(names(groups), times = lapply(groups, length))
  
  x[, ethnicity := names(map)[match(ethnicity, map)]]
  
}

eth_eicu_cb <- function(x, val_var, env) {
  
  x[ethnicity %in% c("Native American", "Other/Unknown"), ethnicity := "Other"]
  
}

