
invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

# needs:
# \usepackage{booktabs}
# \usepackage{longtable}
# \usepackage{pifont}
# \usepackage{makecell}


ynm <- function(x) {
  ifelse(is_true(x), "\\ding{51}",ifelse(is_false(x), "\\ding{55}", "-"))
}

srcs <- c("mimic", "eicu", "hirid", "aumc")

dict <- load_dictionary(srcs)
vars <- read_var_json()

dict <- dict[Filter(Negate(is.na), vars$concept)]

res <- concept_availability(dict, include_rec = FALSE)
res <- cbind(name = rownames(res), as.data.frame(res))

res <- merge(explain_dictionary(dict, cols = c("name", "description")), res,
             by = "name", all.x = TRUE)

res[, srcs] <- lapply(res[, srcs], ynm)
res <- res[, c("name", "description", srcs)]
res$name <- sub("_", "\\_", res$name, fixed = TRUE)
res$description <- kableExtra::linebreak(
  vapply(strwrap(res$description, 35, simplify = FALSE), paste, character(1L),
         collapse = "\n"),
  align = "l"
)
colnames(res) <- c("Name", "Description", "MIMIC-III", "eICU", "HiRID", "AUMC")

kableExtra::kbl(res, "latex", booktabs = TRUE, longtable = TRUE, escape = FALSE,
  caption = "Variables used for sepsis prediction.", label = "variables")
