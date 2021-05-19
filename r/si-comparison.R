library(ricu)
library(RColorBrewer)
library(VennDiagram)

diff_summary <- function(tbl1, tbl2, plt.name, plot.main) {
  tbl1 <- data.table::setorderv(tbl1, cols = c(id_var(tbl1), index_var(tbl1)))
  tbl2 <- data.table::setorderv(tbl2, cols = c(id_var(tbl2), index_var(tbl2)))
  tbl1 <- tbl1[, head(.SD, n = 1L), by = eval(id_var(tbl1))]
  tbl2 <- tbl2[, head(.SD, n = 1L), by = eval(id_var(tbl1))]
  
  A <- unique(tbl1[[id_var(tbl1)]])
  B <- unique(tbl2[[id_var(tbl2)]])
  final <- merge(tbl1, tbl2, by = id_var(tbl1), all = T)
  final[id_col(final) %in% intersect(A, B)]
  C <- final[[index_var(tbl1)]] - final[[index_var(tbl2)]]
  
  
  myCol <- c("#B3E2CD", "#FDCDAC") # brewer.pal(2, "Pastel2")
  
  # Chart
  venn.diagram(x = list(A, B), fill = c("lightblue", "green"), 
               category.names = c("SI + ABX", "multi-ABX"),
               alpha = c(0.5, 0.5), lwd=0.1, 
               height = 1200, 
               width = 1200,
               main.cex = 2,
               cat.pos = c(-30, 40),
               resolution = 300,
               main = plot.main,
               compression = "lzw",
               filename = plt.name)
  
  return(list(A, B, C))
  
}

si_cmp <- function(src) {
  
  beau <- function(src) ifelse(src == "mimic", "MIMIC-III", 
                               ifelse(src == "aumc", "AUMC", "wrong dataset"))
  
  patient_ids <- read_json(file.path(root, "config", "splits",
                                     paste0("splits_", src, ".json")), 
                           simplifyVector = T)[["total"]][["ids"]]
  
  orig <- load_concepts("susp_inf", src, patient_ids = patient_ids)
  orig <- orig[, meta_vars(orig), with = FALSE]
  
  abx <- ricu:::si_abx(load_concepts("abx", src, patient_ids = patient_ids), 
                       hours(24L), 2L) 
  # load_concepts("susp_inf", src, abx_min_count = 2L, si_mode = "or")
  
  abx <- abx[is_true(abx)]
  abx[, abx := NULL]
  abx <- abx[, meta_vars(abx), with = FALSE]
  
  orig <- orig[id_col(orig) %in% patient_ids]
  abx <- abx[id_col(abx) %in% patient_ids]
  
  diff <- diff_summary(orig, abx, paste0("si_comparison_", src, ".tiff"), beau(src))
  
}

si_cmp("mimic")
si_cmp("aumc")
