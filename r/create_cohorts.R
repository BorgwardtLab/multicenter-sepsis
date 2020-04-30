
library(here)
library(ricu)

source(here("r", "utils.R"))

eicu <- sepsis3_score("eicu")
drop <- setdiff(colnames(eicu), id(eicu))
eicu <- unique(eicu[, c("is_septic", drop) := list(TRUE, NULL, NULL)])

hosp <- data_id("eicu", "patient", cols = c("patientunitstayid", "hospitalid"))

dat <- merge(eicu, hosp, all = TRUE)
dat <- dat[, list(septic = sum(is_septic, na.rm = TRUE),
                  total = .N), by = "hospitalid"]
dat <- dat[, prop := septic / total]
dat <- dat[order(prop), ]
dat <- dat[, c("cs_septic", "cs_total") := list(cumsum(septic), cumsum(total))]

print(dat, n = Inf)

dput(dat$hospitalid[dat$prop >= 0.05])
