
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

get_elems <- function(x) {
  if (inherits(x, "rec_cncpt")) {
    x <- lapply(lapply(lapply(x$items, get_elems), as_concept), as.list)
    new_concept(unlist(x, recursive = FALSE))
  } else {
    x
  }
}

srcs <- c("mimic", "eicu", "hirid", "aumc")

dict <- load_dictionary(srcs)
vars <- read_var_json()
vars <- vars[!vars$category %in% c("baseline", "extra"), ]
vars <- cbind(vars, emory = !is.na(vars$callenge))

dict <- dict[Filter(Negate(is.na), vars$concept)]

res <- concept_availability(dict, include_rec = FALSE)
res <- cbind(name = rownames(res), as.data.frame(res))
res <- merge(res, vars[, c("concept", "emory")],
             by.x = "name", by.y = "concept", all.x = TRUE)

res <- merge(explain_dictionary(dict, cols = c("name", "description")), res,
             by = "name", all.x = TRUE)

esrc <- c(srcs, "emory")

res[, esrc] <- lapply(res[, esrc], ynm)
res <- res[, c("name", "description", esrc)]
res$name <- sub("_", "\\_", res$name, fixed = TRUE)
res$description <- kableExtra::linebreak(
  vapply(strwrap(res$description, 35, simplify = FALSE), paste, character(1L),
         collapse = "\n"),
  align = "l"
)
colnames(res) <- c("Name", "Description", "MI-III", "eICU", "HiRID",
                   "AUMC", "Emory")

kableExtra::kbl(res, "latex", booktabs = TRUE, longtable = TRUE, escape = FALSE,
  caption = "Variables used for sepsis prediction.", label = "variables")

vars <- read_var_json()
vars <- vars[is_true(vars$category == "baseline"), ]

dict <- load_dictionary(srcs, vars$concept)

grps <- lapply(dict, get_elems)
grps[["sofa"]] <- dict[["sofa"]]$items
lens <- lengths(grps)
grps <- lapply(grps, function(x) {
  cbind(explain_dictionary(x, cols = c("name", "description")),
        concept_availability(x, include_rec = NA))
})

res <- explain_dictionary(dict, cols = c("name", "description"))
colnames(res) <- c("rec_name", "rec_description")
res <- res[rep(seq_along(grps), lens), ]
res <- cbind(res, do.call(rbind, grps))

res$rec_name <- sub("^sofa_", "s", res$rec_name)
res$name <- sub("^sofa_", "s", res$name)
res$rec_description[grepl("^SOFA ", res$rec_description)] <- "SOFA components"

res[, srcs] <- lapply(res[, srcs], ynm)
res <- res[, c("rec_description", "rec_name", "name", "description", srcs)]
res$rec_description <- paste(
  toupper(substring(res$rec_description, 1,1)),
  substring(res$rec_description, 2),
  sep = ""
)
res$rec_name <- sub("_", "\\_", res$rec_name, fixed = TRUE)
res$name <- sub("_", "\\_", res$name, fixed = TRUE)
res$description <- kableExtra::linebreak(
  vapply(strwrap(res$description, 25, simplify = FALSE), paste, character(1L),
         collapse = "\n"),
  align = "l"
)
res <- res[order(res$rec_description), ]
colnames(res) <- c("", "", "Name", "Description", "MI-III", "eICU", "HiRID",
                   "AUMC")
rownames(res) <- NULL

kableExtra::collapse_rows(
  kableExtra::kbl(
    res, "latex", booktabs = TRUE, longtable = TRUE, escape = FALSE,
    caption = "Variables used for baseline scores", label = "baselines"
  ),
  1:2,
  row_group_label_position = "stack"
)
