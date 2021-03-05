library(ricu)
library(jsonlite)
library(stringr)
library(assertthat)
library(icd)
library(plyr)
library(xtable)

root <- file.path(rprojroot::find_root(".git/index"))
utils_dir <- file.path(root, "r", "utils")
invisible(lapply(list.files(utils_dir, full.names = TRUE), source))
Sys.setenv(RICU_CONFIG_PATH = file.path(root, "config", "ricu-dict"))

src <- c("mimic", "eicu", "hirid", "aumc")
cohort_list <- read_json(file.path(root, "config", "cohorts.json"),                                      simplifyVector = T)
cohorts <- lapply(src, function(x) cohort_list[[x]][["final"]])
names(cohorts) <- src

vars <- list(
  age = list(
    concept = "age",
    callback = med_iqr
  ),
  admission = list(
    concept = "adm",
    callback = tab_design
  ),
  death = list(
    concept = "death",
    callback = percent_fun
  ),
  los_icu = list(
    concept = "los_icu",
    callback = med_iqr
  ),
  los_hosp = list(
    concept = "los_hosp",
    callback = med_iqr
  ),
  gender = list(
    concept = "sex",
    callback = tab_design
  ),
  is_vent = list(
    concept = "is_vent",
    callback = count_percent
  ),
  is_vaso = list(
    concept = "is_vaso",
    callback = count_percent
  ),
  is_abx = list(
    concept = "is_abx",
    callback = count_percent
  ),
  is_si = list(
    concept = "is_si",
    callback = count_percent
  ),
  DM = list(
    concept = "DM",
    callback = count_percent
  ),
  LD = list(
    concept = "LD",
    callback = count_percent
  ),
  CRF = list(
    concept = "CRF",
    callback = count_percent
  ),
  Cancer = list(
    concept = "Cancer",
    callback = count_percent
  ),
  CPD = list(
    concept = "CPD",
    callback = count_percent
  ),
  sofa = list(
    concept = "sofa",
    callback = multi_med_iqr
  )
)

pts_source_sum <- function(source, patient_ids) {

  tbl_list <- list()
  for (var in names(vars)) {
    x <- vars[[var]]

    if (var %in% c("DM", "CPD", "CRF", "LD", "Cancer") & 
        source %in% c("eicu_demo", "hirid", "aumc")) next
    if (var %in% c("admission") & source %in% c("hirid")) next
    
    data <- load_concepts(x[["concept"]], source, 
                          patient_ids = patient_ids, 
                          keep_components = T)

    if (var == "sofa") dat <- data.table::copy(data)

    tbl_list[[var]] <- x[["callback"]](data, patient_ids)
  }

  pts_tbl <- Reduce(rbind,
    lapply(
      tbl_list,
      function(x) data.frame(Reduce(cbind, x))
    )
  )

  cohort_info <- as.data.frame(cbind("Cohort size", "n", 
                                     length(patient_ids)))

  si <- load_concepts("susp_inf_mc", source, patient_ids = patient_ids)
  si <- rename_cols(si, "susp_inf", "susp_inf_mc")
  s3_info <- sep3(dat, si, si_window = "any")
  s3_cohort <- unique(id_col(s3_info[sep3 == T]))

  s3_entry <- paste0(
    length(s3_cohort), " (",
    round(100 *  length(s3_cohort) / length(patient_ids), 2), ")"
  )

  cohort_info <- rbind(
    cohort_info,
    cbind("Sepsis-3 prevalence", "n (%)", V3 = s3_entry)
  )
  names(cohort_info) <- names(pts_tbl)

  pts_tbl <- rbind(
    cohort_info,
    pts_tbl
  )

  names(pts_tbl) <- c("Variable", "Reported", srcwrap(source))

  pts_tbl$Variable <- mapvalues(pts_tbl$Variable,
    from = names(concept_translator),
    to = sapply(names(concept_translator), function(x) concept_translator[[x]])
  )

  pts_tbl

}

res <- Reduce(
  function(x, y) merge(x, y, by = c("Variable", "Reported"), 
                       sort = F, all = T),
  Map(pts_source_sum, src, cohorts)
)

print(xtable(res, align = c("c", "l", "c", "c", "c", "c", "c")), 
      floating=FALSE, latex.environments=NULL, include.rownames = F)

