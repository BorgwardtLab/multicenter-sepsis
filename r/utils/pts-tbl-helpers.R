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

clean_unit_str <- function(x) {
  x <- gsub("/l", "/L", x)
  x <- gsub("/dl", "/dL", x)
  x <- gsub("mmhg", "mmHg", x)
  x <- gsub("meq", "mEq", x)
  x <- gsub("\\(iu", "\\(IU", x)
  x
}

nnquant <- function(x, y, upto = hours(24L)) {

  x <- x[get(index_var(x)) <= upto]
  val_col <- setdiff(names(x), meta_vars(x))
  x <- x[[val_col]]

  med <- round(median(x, na.rm = T), 2)
  IQR <- quantile(x, c(0.25, 0.75), na.rm = T)
  n <- length(x)

  res <- paste0(med, " (", round(IQR[1], 2), "-", round(IQR[2], 2), ")")

  c(
    "Number,\n Number per patient stay,\n Median (IQR)",
    paste(n, round(sum(!is.na(x))/length(y), 2), res, sep = ",\n")
  )

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

count_percent <- function(x, patient_ids) {

  val_col <- setdiff(names(x), meta_vars(x))
  list(val_col,
    "n (%)",
    paste0(
      sum(x[[val_col]]), " (",
      round(100 * sum(x[[val_col]]) / length(patient_ids)), ")"
    )
  )

}

concept_translator <- list(
  age = "Age, years",
  med = "hspace3mm Medical",
  other = "hspace3mm Other",
  surg = "hspace3mm Surgical",
  death = "In-hospital mortality",
  `Cohort size` = "Cohort size",
  los_icu = "ICU LOS, days",
  los_hosp = "Hospital LOS, days",
  Male = "Gender, male",
  Female = "Gender, female",
  sofa = "Initial SOFA",
  sofa_resp_comp = "hspace3mm Respiratory",
  sofa_coag_comp = "hspace3mm Coagulation",
  sofa_cns_comp = "hspace3mm CNS",
  sofa_liver_comp = "hspace3mm Hepatic",
  sofa_cardio_comp = "hspace3mm Cardiovascular",
  sofa_renal_comp = "hspace3mm Renal",
  is_vaso = "Patients on vasopressors",
  is_abx = "Patients on antibiotics",
  is_vent = "Ventilated patients",
  is_si = "Patients with suspected infection",
  DM = "hspace3mm Diabetes",
  LD = "hspace3mm Liver Disease",
  CPD = "hspace3mm Chronic Pulmonary Disease",
  CRF = "hspace3mm Chronic Renal Failure",
  Cancer = "hspace3mm Cancer"
)

pts_split <- function(src, split = "total") {
  assert_that(split %in% c("dev", "test", "total"),
              msg = "Unknown split requested.")
  if (src == "eicu_demo") return(eicu_demo$patient$patientunitstayid)
  fl <- read_json(file.path(root, "config", "splits",
                            paste0("splits_", src, ".json")), 
                  simplifyVector = T)
  if (split == "total") return(fl[[split]][["ids"]])
  fl[[split]][["total"]]
}
