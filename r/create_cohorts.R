#!/usr/bin/env Rscript

#BSUB -W 1:00
#BSUB -n 1
#BSUB -R rusage[mem=8000]
#BSUB -J cohorts
#BSUB -o results/cohorts_%J.out

lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)

eicu_hospitals <- function(thresh = 0.05) {

  stopifnot(is.numeric(thresh), length(thresh) == 1L, thresh > 0, thresh < 1)

  eicu <- sepsis3_crit("eicu")
  eicu <- unique(eicu[, c(index_var(eicu)) := NULL])

  hosp <- load_id("patient", "eicu",
                  cols = c("patientunitstayid", "hospitalid"))

  dat <- merge(eicu, hosp, all = TRUE)
  dat <- dat[, list(septic = sum(sepsis_3, na.rm = TRUE),
                    total = .N), by = "hospitalid"]
  dat <- dat[, prop := septic / total]

  res <- unique(dat$hospitalid[dat$prop >= thresh])

  message(
    "* selecting ", length(res), " hospitals from ",
    length(unique(hosp$hospitalid)), " based on a sep3 prevalence of ", thres,
    " "
  )

  res
}

cohort <- function(source, min_age = 14) {

  stopifnot(is.numeric(min_age), length(min_age) == 1L)

  message("determining cohort for ", source)

  res <- load_concepts("age", source, id_type = "icustay")
  nrw <- nrow(res)
  res <- res[age > min_age, ]

  message("* removing ", nrw - nrow(res), " from ", nrw,
          " ids due to min age of ", min_age)

  if (grepl("eicu", source)) {
    hosp <- load_id("patient", source, cols = c(id_var(res), "hospitalid"))
    hosp <- hosp[hospitalid %in% eicu_hospitals(), ]
    nrw <- nrow(res)
    res <- merge(res, hosp)
    message("* removing ", nrw - nrow(res), " from ", nrw,
            " ids based on hosp selection")
  }

  id_col(res)
}

src <- check_src(parse_args(src_opt))
res <- lapply(src, cohort)
names(res) <- src

jsonlite::write_json(res, cfg_path("cohorts.json"))
