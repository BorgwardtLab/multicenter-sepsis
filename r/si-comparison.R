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
  venn.diagram(
    x = list(A, B),
    category.names = c("SI + ABX" , "ABX"),
    filename = plt.name,
    output=TRUE,
    main = plot.main,
    
    # Output features
    imagetype="png" ,
    height = 600 , 
    width = 800 , 
    resolution = 300,
    compression = "lzw",
    
    # Circles
    lwd = 2,
    lty = 'blank',
    fill = myCol,
    
    # Numbers
    cex = .6,
    fontface = "bold",
    fontfamily = "sans",
    
    # Set names
    cat.cex = 0.6,
    cat.fontface = "bold",
    cat.default.pos = "outer",
    cat.pos = c(-27, 27),
    cat.dist = c(0.055, 0.055),
    cat.fontfamily = "sans"
  )
  
  return(list(A, B, C))
  
}

si_cmp <- function(src, min.age = 14L) {
  
  beau <- function(src) ifelse(src == "mimic", "MIMIC-III", ifelse(src == "aumc", "AUMC", "wrong dataset"))
  
  orig <- load_concepts("susp_inf", src)
  orig <- orig[, meta_vars(orig), with = FALSE]
  
  abx <- ricu:::si_abx(load_concepts("abx", src), hours(24L), 2L) 
  # load_concepts("susp_inf", src, abx_min_count = 2L, si_mode = "or")
  
  abx <- abx[is_true(abx)]
  abx[, abx := NULL]
  abx <- abx[, meta_vars(abx), with = FALSE]
  
  adults <- id_col(load_concepts("age", src)[age > min.age])
  orig <- orig[id_col(orig) %in% adults]
  abx <- abx[id_col(abx) %in% adults]
  
  diff <- diff_summary(orig, abx, paste0("si_comparison_", src, ".png"), beau(src))
  
}

si_cmp("mimic")
si_cmp("aumc")
