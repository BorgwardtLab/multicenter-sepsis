library(ricu)
library(stringr)
library(magrittr)
library(officer)
library(assertthat)
library(plyr)

source("pts-tbl-helpers.R")

cohort_gen <- function(src) id_col(load_concepts("susp_inf", src, si_mode = "or"))

src <- c("mimic_demo", "eicu_demo") # c("mimic", "eicu", "hirid", "aumc")
cohorts <- lapply(src, cohort_gen)

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
  sofa = list(
    concept = "sofa",
    callback = multi_med_iqr)
  # ),
  # sep3 = list(
  #   concept = "sep3",
  #   callback = percent_fun
  # )
)

pts_source_sum <- function(source, patient_ids) {

  tbl_list <- lapply(
    vars,
    function(x) x[["callback"]](
      load_concepts(x[["concept"]], source, patient_ids = patient_ids, keep_components = T), patient_ids
      )
  )

  pts_tbl <- Reduce(rbind,
    lapply(
      tbl_list,
      function(x) data.frame(Reduce(cbind, x))
    )
  )

  cohort_info <- as.data.frame(cbind("Cohort size", "n", length(patient_ids)))
  cohort_info <- rbind(cohort_info, 
    cbind("Sepsis-3 prevalence", "%", round(100 * sum(s3_pts_tbl("mimic_demo")[["sep3"]]) / length(patient_ids), 2)))
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
  function(x, y) merge(x, y, by = c("Variable", "Reported"), sort = F),
  Map(pts_source_sum, src, cohorts)
)

my_doc <- read_docx()

my_doc <- my_doc %>%
  body_add_table(res, style = "table_template")

print(my_doc, target = "Table1.docx")