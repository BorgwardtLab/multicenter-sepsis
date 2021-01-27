srcwrap <- function(src) {
  if (length(src) > 1) return(sapply(src, srcwrap))

  if(src == "mimic") {
    return("MIMIC-III")
  } else if (src == "eicu") {
    return("eICU")
  } else if (src == "hirid") {
    return("HiRID")
  } else if (src == "mimic_demo") {
    return("MIMIC-III (demo)")
  } else if (src == "eicu_demo") {
    return("eICU (demo)")
  } else if (src == "aumc") {
    return("AUMC")
  } else {
    return(src)
  }
}

undemo <- function(src) {

  if(src == "mimic_demo") return("mimic")
  if(src == "eicu_demo") return("eicu")

  return(src)

}


med_iqr <- function(x, patient_ids) {
  val_col <- setdiff(names(x), meta_vars(x))
  if(is_ts_tbl(x)) x <- x[get(index_var(x)) == 24L]
  quants <- quantile(x[[val_col]], probs = c(0.25, 0.5, 0.75), na.rm = T)
  res <- paste0(
    round(quants[2], 2), " (",
    round(quants[1], 2), "-",
    round(quants[3], 2), ")"
  )

  list(val_col, "Median (IQR)", res)
}

multi_med_iqr <- function(x, patient_ids) {

  val_cols <- setdiff(names(x), meta_vars(x))
  res <- lapply(
    val_cols, function(vcol) med_iqr(x[, c(meta_vars(x), vcol), with = FALSE], patient_ids)
  )

  lapply(1:3, function(i) {
    Reduce(c, lapply(res, `[[`, i))
  })

}

tab_design <- function(x, patient_ids) {

  val_col <- setdiff(names(x), meta_vars(x))
  res <- table(x[[val_col]])
  res <- round(100 * res / sum(res))

  if(val_col == "adm" & nrow(x) == 0L) {

    return(
      list(c("med", "surg", "other"), "%", rep(NA, 3))
    )

  }

  list(names(res), "%", as.integer(res))

}

percent_fun <- function(x, patient_ids) {

  val_col <- setdiff(names(x), meta_vars(x))
  list(val_col, "%", round(100 * sum(x[[val_col]]) / length(patient_ids)))

}

concept_translator <- list(
  age = "Age (years)",
  med = "- Medical",
  other = "- Other",
  surg = "- Surgical",
  death = "Mortality",
  `Cohort size` = "Cohort size",
  los_icu = "ICU LOS",
  los_hosp = "Hospital LOS (days)",
  Male = "Gender (Male)",
  Female = "Gender (Female)",
  sofa = "- Total",
  sofa_resp_comp = "- Respiratory",
  sofa_coag_comp = "- Coagulation",
  sofa_cns_comp = "- CNS",
  sofa_liver_comp = "- Hepatic",
  sofa_cardio_comp = "- Cardiovascular",
  sofa_renal_comp = "- Renal",
  sep3 = "Sepsis-3 prevalence"
)

s3_pts_tbl <- function(source, pids = NULL) {

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

  sep3(load_concepts("sofa", source), si)
}
