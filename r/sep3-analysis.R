
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
  
  if (src %in% c("eicu", "hirid")) next
  
  tbl <- sep[[src]]
  
  tbl <- tbl[, class := (get(index_var(tbl)) < samp_time) + 
               (get(index_var(tbl)) < abx_time)]
  
  res <- table(tbl[["class"]])
  res <- 100 * res / sum(res)
  names(res) <- c("Delta last", "Delta middle", "Delta first")
  
  print(res)
  
}


