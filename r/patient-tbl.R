library(ricu)
library(jsonlite)
library(stringr)
library(assertthat)
library(icd)
library(plyr)
library(xtable)
library(data.table)
library(flextable)

root <- file.path(rprojroot::find_root(".git/index"))
utils_dir <- file.path(root, "r", "utils")
invisible(lapply(list.files(utils_dir, full.names = TRUE), source))
Sys.setenv(RICU_CONFIG_PATH = file.path(root, "config", "ricu-dict"))

src <- c("mimic", "eicu", "hirid", "aumc")
emory <- TRUE
vars <- list(
  age = list(
    concept = "age",
    callback = med_iqr
  ),
  admission = list(
    concept = "adm",
    callback = tab_design
  ),
  ethnicity = list(
    concept = "ethnicity",
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

for (split in c("test")) {
  cohorts <- lapply(src, pts_split, split = split)
  names(cohorts) <- src
  
  pts_source_sum <- function(source, patient_ids) {
    
    tbl_list <- list()
    for (var in names(vars)) {
      x <- vars[[var]]
      
      if (var %in% c("DM", "CPD", "CRF", "LD", "Cancer", "ethnicity") &
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
    
    pts_tbl$V3 <- gsub("NA \\(NA-NA\\)", "-", pts_tbl$V3)
    
    
    names(pts_tbl) <- c("Variable", "Reported", srcwrap(source))
    
    pts_tbl$Variable <- mapvalues(
      pts_tbl$Variable,
      from = names(concept_translator),
      to = sapply(names(concept_translator), \(x) concept_translator[[x]])
    )
    
    add_report <- ifelse(grepl("hspace3mm", pts_tbl$Variable),
                         "", paste0(" (", pts_tbl$Reported, ")"))
    
    pts_tbl$Variable <- paste0(pts_tbl$Variable, add_report)
    pts_tbl$Variable <- gsub("hspace3mm", "    ", pts_tbl$Variable)
    pts_tbl$Reported <- NULL
    
    pts_tbl
    
  }
  
  res <- Reduce(
    function(x, y) merge(x, y, by = c("Variable"),
                         sort = F, all = T),
    Map(pts_source_sum, src, cohorts)
  )
  
  # make columns into characters
  for (col in names(res)) res[[col]] <- as.character(res[[col]])
  
  # replace NAs by "-"
  rep_list <- lapply(seq_along(names(res)), function(x) "-")
  names(rep_list) <- names(res)
  res <- tidyr::replace_na(res, rep_list)
  
  if (emory) {
    
    res <- cbind(res, Emory = "-", stringsAsFactors = FALSE)
    
    dat <- arrow::read_parquet(file.path(root, "emory-data", 
                                         "physionet2019_0.4.0.parquet"))
    dat <- as.data.table(dat)
    
    # subset on the case
    ids <- pts_split("physionet2019", split = split)
    dat <- dat[stay_id %in% ids]
    # cohort size
    n_coh <- length(unique(dat$stay_id))
    res[res$Variable == "Cohort size (n)", "Emory"] <- n_coh
    
    # prevalence
    dat$onset_ind
    n_sep <- sum(dat[, any(onset_ind), by = "stay_id"][["V1"]], na.rm = T)
    perc_sep <- round(n_sep / n_coh * 100, 2)
    res[res$Variable == "Sepsis-3 prevalence (n (%))", "Emory"] <- 
      paste0(n_sep, " (", perc_sep, ")")
    
    # age
    age <- as_id_tbl(dat[, unique(age_static), by = "stay_id"], 
                     id_vars = "stay_id")
    
    res[res$Variable == "Age, years (Median (IQR))", "Emory"] <- 
      med_iqr(age, NULL)[[3L]]
    
    # gender
    n_fem <- sum(dat[, any(female_static), by = "stay_id"][["V1"]])
    perc_fem <- round(n_fem / n_coh * 100)
    res[res$Variable == "Gender, female (%)", "Emory"] <- perc_fem
    res[res$Variable == "Gender, male (%)", "Emory"] <- 100L - perc_fem
    
  }
  
  ftab <- flextable(res)
  ftab <- autofit(ftab)
  save_as_docx(ftab, path = file.path(root, "results", 
                                      paste0("patient_table_", split, ".docx"))) 
}
