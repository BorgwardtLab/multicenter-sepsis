invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

vars <- jsonlite::read_json(cfg_path("variables.json"))
vars <- unlist(lapply(vars, `[[`, "concept"))
rm_vars <- c("sofa_resp", "sofa_coag", "sofa_liver", "sofa_cardio", 
             "sofa_cns", "sofa_renal", "sofa", "qsofa", "sirs", "news", "mews", 
             "death", "abx", "sex")
vars <- vars[!(vars %in% rm_vars)]
dict <- get_config("concept-dict")

wrap <- function(x) if (is.null(x)) return("-") else x

tab <- lapply(
  vars,
  \(x) {
    c(x, wrap(dict[[x]]$description), wrap(dict[[x]]$unit[1]), 
      wrap(dict[[x]]$min), wrap(dict[[x]]$max))
  }
)

tab <- as.data.frame(Reduce(rbind, tab))
names(tab) <- c("Name", "Description", "Unit of meas.", "Min. value", 
                "Max. value")
save_as_docx(autofit(flextable(tab)), path = "results/units_and_ranges.docx")
