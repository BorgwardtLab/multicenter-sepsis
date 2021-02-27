#!/usr/bin/env Rscript

#BSUB -W 1:00
#BSUB -n 1
#BSUB -R rusage[mem=24000]
#BSUB -J cohorts
#BSUB -o results/cohorts_%J.out

invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

eicu_hospitals <- function(thresh = 0.05) {

  assert(is.numeric(thresh), is.scalar(thresh), thresh > 0, thresh < 1)

  eicu <- sepsis3_crit("eicu")
  eicu <- unique(eicu[, c(index_var(eicu)) := NULL])

  hosp <- load_id("patient", "eicu",
                  cols = c("patientunitstayid", "hospitalid"))

  dat <- merge(eicu, hosp, all = TRUE)
  dat <- dat[, list(septic = sum(sep3, na.rm = TRUE),
                    total = .N), by = "hospitalid"]
  dat <- dat[, prop := septic / total]

  res <- unique(dat$hospitalid[dat$prop >= thresh])

  msg("\n\n--> selecting {length(res)} hospitals from",
      " {length(unique(hosp$hospitalid))} based on a sep3 prevalence of",
      " {thresh}\n")

  res
}

cohort <- function(source, min_age = 14) {

  assert(is.count(min_age))

  msg("\n\ndetermining cohort for {source}\n\n")

  res <- load_concepts("age", source, id_type = "icustay")
  nrw <- nrow(res)
  res <- res[age > min_age, ]

  msg("\n\n--> removing {nrw - nrow(res)} from {nrw} ids due to min age of",
      " {min_age}\n")

  out <- list(initial = unique(id_col(res)))

  if (grepl("eicu", source)) {
    hos <- load_id("patient", source, cols = c(id_var(res), "hospitalid"))
    hos <- hos[hospitalid %in% eicu_hospitals(), ]
    nrw <- nrow(res)
    res <- merge(res, hos)
    msg("\n\n--> removing {nrw - nrow(res)} from {nrw} ids based on hosp",
        " selection\n")
    out[["hospitals"]] <- unique(res$hospitalid)
  }

  out
}

src <- check_src(parse_args(src_opt))
res <- lapply(src, cohort)

names(res) <- src

jsonlite::write_json(res, cfg_path("cohorts.json"), pretty = TRUE)
