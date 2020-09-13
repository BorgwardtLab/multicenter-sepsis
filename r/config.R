
concepts <- list(
      list("hr", "HR", readr::col_double())
    , list("o2sat", "O2Sat", readr::col_double())
    , list("temp", "Temp", readr::col_double())
    , list("sbp", "SBP", readr::col_double())
    , list("map", "MAP", readr::col_double())
    , list("dbp", "DBP", readr::col_double())
    , list("resp", "Resp", readr::col_double())
    , list("etco2", "EtCO2", readr::col_double())
    , list("be", "BaseExcess", readr::col_double())
    , list("bicar", "HCO3", readr::col_double())
    , list("fio2", "FiO2", readr::col_double())
    , list("ph", "pH", readr::col_double())
    , list("pco2", "PaCO2", readr::col_double())
    , list(NA_character_, "SaO2", readr::col_double())
    , list("ast", "AST", readr::col_double())
    , list("bun", "BUN", readr::col_double())
    , list("alp", "Alkalinephos", readr::col_double())
    , list("ca", "Calcium", readr::col_double())
    , list("cl", "Chloride", readr::col_double())
    , list("crea", "Creatinine", readr::col_double())
    , list("bili_dir", "Bilirubin_direct", readr::col_double())
    , list("glu", "Glucose", readr::col_double())
    , list("lact", "Lactate", readr::col_double())
    , list("mg", "Magnesium", readr::col_double())
    , list("phos", "Phosphate", readr::col_double())
    , list("k", "Potassium", readr::col_double())
    , list("bili", "Bilirubin_total", readr::col_double())
    , list("tri", "TroponinI", readr::col_double())
    , list("hct", "Hct", readr::col_double())
    , list("hgb", "Hgb", readr::col_double())
    , list("ptt", "PTT", readr::col_double())
    , list("wbc", "WBC", readr::col_double())
    , list("fgn", "Fibrinogen", readr::col_double())
    , list("plt", "Platelets", readr::col_double())
    , list("age", "Age", readr::col_integer())
    , list("sex", "Gender", readr::col_integer())
    , list(NA_character_, "Unit1", readr::col_skip())
    , list(NA_character_, "Unit2", readr::col_skip())
    , list(NA_character_, "HospAdmTime", readr::col_skip())
    , list(NA_character_, "ICULOS", readr::col_integer())
    , list(NA_character_, "SepsisLabel", readr::col_logical())
    , list("alb", NA_character_, NULL)
    , list("alt", NA_character_, NULL)
    , list("basos", NA_character_, NULL)
    , list("bnd", NA_character_, NULL)
    , list("cai", NA_character_, NULL)
    , list("ck", NA_character_, NULL)
    , list("ckmb", NA_character_, NULL)
    , list("crp", NA_character_, NULL)
    , list("eos", NA_character_, NULL)
    , list("esr", NA_character_, NULL)
    , list("hbco", NA_character_, NULL)
    , list("inr_pt", NA_character_, NULL)
    , list("lymph", NA_character_, NULL)
    , list("mch", NA_character_, NULL)
    , list("mchc", NA_character_, NULL)
    , list("mcv", NA_character_, NULL)
    , list("methb", NA_character_, NULL)
    , list("na", NA_character_, NULL)
    , list("neut", NA_character_, NULL)
    , list("po2", NA_character_, NULL)
    , list("pt", NA_character_, NULL)
    , list("rbc", NA_character_, NULL)
    , list("rdw", NA_character_, NULL)
    , list("tco2", NA_character_, NULL)
    , list("tnt", NA_character_, NULL)
    , list("sirs", NA_character_, NULL)
    , list("news", NA_character_, NULL)
    , list("mews", NA_character_, NULL)
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
