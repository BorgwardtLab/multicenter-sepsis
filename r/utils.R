
challenge_map <- data.frame(rbind(
    c("HR", "heart_rate")
  , c("O2Sat", "o2_saturation")
  , c("Temp", "temperature")
  , c("SBP", "systolic_bp")
  , c("MAP", "mean_bp")
  , c("DBP", "diastolic_bp")
  , c("Resp", "respiratory_rate")
  , c("EtCO2", "et_co2")
  , c("BaseExcess", "base_excess")
  , c("HCO3", "bicarbonate")
  , c("FiO2", "fi_o2")
  , c("pH", "ph")
  , c("PaCO2", "pa_co2")
# , c("SaO2", "?")
  , c("AST", "asparate_aminotransferase")
  , c("BUN", "urea_nitrogen")
  , c("Alkalinephos", "alkaline_phosphatase")
  , c("Calcium", "calcium")
  , c("Chloride", "chloride")
  , c("Creatinine", "creatinine")
  , c("Bilirubin_direct", "bilirubin_direct")
  , c("Glucose", "glucose")
  , c("Lactate", "lactate")
  , c("Magnesium", "magnesium")
  , c("Phosphate", "phosphate")
  , c("Potassium", "potassium")
  , c("Bilirubin_total", "bilirubin_total")
  , c("TroponinI", "troponin_i")
  , c("Hct", "hematocrit")
  , c("Hgb", "hemoglobin")
  , c("PTT", "ptt")
  , c("WBC", "white_blood_cells")
  , c("Fibrinogen", "fibrinogen")
  , c("Platelets", "platelet_count")
  , c("Age", "age")
  , c("Gender", "sex")
# , c("Unit1", "?")
# , c("Unit2", "?")
# , c("HospAdmTime", "?")
# , c("ICULOS", "?")
# , c("SepsisLabel", "?")
) , stringsAsFactors = FALSE)

sepsis3_score <- function(source, pids = NULL) {

  sofa <- sofa_data(source, id_type = "icustay", patient_ids = pids)
  sofa <- sofa_window(sofa)
  sofa <- sofa_compute(sofa)

  if (grepl("eicu", source)) {

    suin <- si_data(source, abx_min_count = 2L, positive_cultures = TRUE,
                    id_type = "icustay", patient_ids = pids)
    suin <- si_windows(suin, mode = "or")

  } else if (identical(source, "hirid")) {

    suin <- si_data(source, abx_min_count = 2L, id_type = "icustay",
                    patient_ids = pids)
    suin <- si_windows(suin, mode = "or")

  } else {

    suin <- si_data(source, id_type = "icustay", patient_ids = pids)
    suin <- si_windows(suin)
  }

  sepsis_3(sofa, suin)
}

dump_dataset <- function(source = "mimic_demo", dir = tempdir()) {

  dat <- load_dictionary(source, challenge_map[[2L]], id_type = "icustay")
  ids <- unique(dat[age >= 14, id(dat), with = FALSE])
  dat <- merge(dat, ids, all.y = TRUE)

  sep3 <- sepsis3_score(source, ids)
  sep3 <- data.table::set(sep3, j = "SepsisLabel", value = 1L)

  dat <- rename_cols(dat, challenge_map[[1L]], challenge_map[[2L]],
                     skip_absent = TRUE)

  res <- merge(dat, sep3[, c(meta_cols(sep3), "SepsisLabel"), with = FALSE],
               all.x = TRUE)
  res <- res[, c("SepsisLabel") := data.table::nafill(SepsisLabel, "locf"),
             by = c(id(res))]
  res <- res[, c("SepsisLabel") := data.table::nafill(SepsisLabel, fill = 0L)]

  dir <- file.path(dir, source)

  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE)

  write_psv(res, dir, na_rm = TRUE)
}
