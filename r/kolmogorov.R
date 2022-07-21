
library(flextable)
invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

Sys.setenv(RICU_CONFIG_PATH = file.path(here::here(), "config", "ricu-dict"))

cncs <- c("lact", "crea", "ast")
res <- list()
for (cnc in cncs) {
  
  tbl <- load_concepts(c(cnc, "careunit"), "mimic", verbose = FALSE)
  
  tbl <- tbl[!is.na(get(cnc)), head(.SD, n = 1L), by = c(id_vars(tbl))]
  
  micu <- tbl[careunit == "MICU"][[cnc]]
  micu <- micu + rnorm(length(micu), sd = 0.001)
  
  oicu <- tbl[careunit != "MICU"][[cnc]]
  oicu <- oicu + rnorm(length(oicu), sd = 0.001)
  
  tst <- ks.test(x = micu, y = oicu)
  
  res[[cnc]] <- list(
    concept = cnc,
    itemid = get_config("concept-dict")[[cnc]][["sources"]][["mimic"]][[1]]$ids,
    Dstat = round(tst$statistic, 3),
    pval = round(tst$p.value, 5),
    nsamp1 = length(micu),
    nsamp2 = length(oicu)
  )
  
}

tab <- Reduce(rbind, lapply(res, as.data.frame))
names(tab) <- c("Variable", "MIMIC-III itemid", "D statistic", "p-value", 
                "Sample size (MICU)", "Sample size (not MICU)")

save_as_docx(autofit(flextable(tab)), path = "results/kolmogorov.docx")
