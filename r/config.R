
concepts <- list(
      list("heart_rate", "HR", readr::col_double())
    , list("o2_saturation", "O2Sat", readr::col_double())
    , list("temperature", "Temp", readr::col_double())
    , list("systolic_bp", "SBP", readr::col_double())
    , list("mean_bp", "MAP", readr::col_double())
    , list("diastolic_bp", "DBP", readr::col_double())
    , list("respiratory_rate", "Resp", readr::col_double())
    , list("et_co2", "EtCO2", readr::col_double())
    , list("base_excess", "BaseExcess", readr::col_double())
    , list("bicarbonate", "HCO3", readr::col_double())
    , list("fi_o2", "FiO2", readr::col_double())
    , list("ph", "pH", readr::col_double())
    , list("pa_co2", "PaCO2", readr::col_double())
    , list(NA_character_, "SaO2", readr::col_double())
    , list("asparate_aminotransferase", "AST", readr::col_double())
    , list("urea_nitrogen", "BUN", readr::col_double())
    , list("alkaline_phosphatase", "Alkalinephos", readr::col_double())
    , list("calcium", "Calcium", readr::col_double())
    , list("chloride", "Chloride", readr::col_double())
    , list("creatinine", "Creatinine", readr::col_double())
    , list("bilirubin_direct", "Bilirubin_direct", readr::col_double())
    , list("glucose", "Glucose", readr::col_double())
    , list("lactate", "Lactate", readr::col_double())
    , list("magnesium", "Magnesium", readr::col_double())
    , list("phosphate", "Phosphate", readr::col_double())
    , list("potassium", "Potassium", readr::col_double())
    , list("bilirubin_total", "Bilirubin_total", readr::col_double())
    , list("troponin_i", "TroponinI", readr::col_double())
    , list("hematocrit", "Hct", readr::col_double())
    , list("hemoglobin", "Hgb", readr::col_double())
    , list("ptt", "PTT", readr::col_double())
    , list("white_blood_cells", "WBC", readr::col_double())
    , list("fibrinogen", "Fibrinogen", readr::col_double())
    , list("platelet_count", "Platelets", readr::col_double())
    , list("age", "Age", readr::col_integer())
    , list("sex", "Gender", readr::col_integer())
    , list(NA_character_, "Unit1", readr::col_skip())
    , list(NA_character_, "Unit2", readr::col_skip())
    , list(NA_character_, "HospAdmTime", readr::col_skip())
    , list(NA_character_, "ICULOS", readr::col_integer())
    , list(NA_character_, "SepsisLabel", readr::col_logical())
    , list("sirs_score", NA_character_, NULL)
    , list("news_score", NA_character_, NULL)
    , list("mews_score", NA_character_, NULL)
)

concepts <- data.frame(
  concept = vapply(concepts, `[[`, character(1L), 1L),
  callenge = vapply(concepts, `[[`, character(1L), 2L),
  col_spec = I(lapply(concepts, `[[`, 3L))
)

eicu_hospitals <- c(
   56L,  58L,  59L,  60L,  61L,  63L,  66L,  67L,  68L,  69L,  71L,  73L,
   79L,  83L,  84L,  86L,  91L,  92L,  93L, 146L, 148L, 151L, 152L, 154L,
  155L, 157L, 158L, 164L, 165L, 167L, 171L, 174L, 175L, 176L, 180L, 181L,
  182L, 183L, 184L, 194L, 195L, 198L, 199L, 200L, 202L, 206L, 212L, 215L,
  217L, 220L, 224L, 226L, 227L, 243L, 244L, 245L, 248L, 249L, 250L, 251L,
  252L, 253L, 259L, 264L, 268L, 269L, 271L, 272L, 273L, 275L, 277L, 279L,
  280L, 281L, 282L, 283L, 300L, 303L, 307L, 310L, 312L, 323L, 328L, 331L,
  336L, 337L, 338L, 342L, 345L, 353L, 358L, 360L, 364L, 365L, 420L, 421L,
  422L, 423L, 424L, 425L, 428L, 429L, 433L, 434L, 435L, 436L, 437L, 438L,
  439L, 440L, 443L, 445L, 447L, 449L, 459L
)
