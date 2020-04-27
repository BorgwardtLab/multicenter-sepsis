
challenge_map <- data.frame(
  rbind(
      c("HR",               "heart_rate")
    , c("O2Sat",            "o2_saturation")
    , c("Temp",             "temperature")
    , c("SBP",              "systolic_bp")
    , c("MAP",              "mean_bp")
    , c("DBP",              "diastolic_bp")
    , c("Resp",             "respiratory_rate")
    , c("EtCO2",            "et_co2")
    , c("BaseExcess",       "base_excess")
    , c("HCO3",             "bicarbonate")
    , c("FiO2",             "fi_o2")
    , c("pH",               "ph")
    , c("PaCO2",            "pa_co2")
  # , c("SaO2",             "?")
    , c("AST",              "asparate_aminotransferase")
    , c("BUN",              "urea_nitrogen")
    , c("Alkalinephos",     "alkaline_phosphatase")
    , c("Calcium",          "calcium")
    , c("Chloride",         "chloride")
    , c("Creatinine",       "creatinine")
    , c("Bilirubin_direct", "bilirubin_direct")
    , c("Glucose",          "glucose")
    , c("Lactate",          "lactate")
    , c("Magnesium",        "magnesium")
    , c("Phosphate",        "phosphate")
    , c("Potassium",        "potassium")
    , c("Bilirubin_total",  "bilirubin_total")
    , c("TroponinI",        "troponin_i")
    , c("Hct",              "hematocrit")
    , c("Hgb",              "hemoglobin")
    , c("PTT",              "ptt")
    , c("WBC",              "white_blood_cells")
    , c("Fibrinogen",       "fibrinogen")
    , c("Platelets",        "platelet_count")
    , c("Age",              "age")
    , c("Gender",           "sex")
  # , c("Unit1",            "?")
  # , c("Unit2",            "?")
  # , c("HospAdmTime",      "?")
  # , c("ICULOS",           "?")
  # , c("SepsisLabel",      "?")
  ),
  stringsAsFactors = FALSE
)

challenge_spec <- readr::cols(
  HR = readr::col_double(),
  O2Sat = readr::col_double(),
  Temp = readr::col_double(),
  SBP = readr::col_double(),
  MAP = readr::col_double(),
  DBP = readr::col_double(),
  Resp = readr::col_double(),
  EtCO2 = readr::col_double(),
  BaseExcess = readr::col_double(),
  HCO3 = readr::col_double(),
  FiO2 = readr::col_double(),
  pH = readr::col_double(),
  PaCO2 = readr::col_double(),
  SaO2 = readr::col_double(),
  AST = readr::col_double(),
  BUN = readr::col_double(),
  Alkalinephos = readr::col_double(),
  Calcium = readr::col_double(),
  Chloride = readr::col_double(),
  Creatinine = readr::col_double(),
  Bilirubin_direct = readr::col_double(),
  Glucose = readr::col_double(),
  Lactate = readr::col_double(),
  Magnesium = readr::col_double(),
  Phosphate = readr::col_double(),
  Potassium = readr::col_double(),
  Bilirubin_total = readr::col_double(),
  TroponinI = readr::col_double(),
  Hct = readr::col_double(),
  Hgb = readr::col_double(),
  PTT = readr::col_double(),
  WBC = readr::col_double(),
  Fibrinogen = readr::col_double(),
  Platelets = readr::col_double(),
  Age = readr::col_integer(),
  Gender = readr::col_integer(),
  Unit1 = readr::col_integer(),
  Unit2 = readr::col_integer(),
  HospAdmTime = readr::col_double(),
  ICULOS = readr::col_integer(),
  SepsisLabel = readr::col_logical()
)
