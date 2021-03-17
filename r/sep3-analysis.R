
invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

demo <- c("mimic_demo", "eicu_demo")
prod <- c("mimic", "eicu", "hirid", "aumc")

srcs <- prod

cohorts <- jsonlite::read_json(
  cfg_path("cohorts.json"), simplifyVector = TRUE, flatten = TRUE
)

cohorts <- lapply(cohorts[srcs], `[[`, "final")

sep <- vector("list", length(srcs))
names(sep) <- srcs

for (src in srcs) {
  sep[[src]] <- sepsis3_crit(src, pids = cohorts[[src]],
                             keep_components = TRUE)
}
