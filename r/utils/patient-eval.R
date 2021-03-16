# dat <- read_res("aumc", feat_set = "basic", predictor = "rf",
#                 target = "class", jobid = 165221956L)

patient_eval <- function(dat) {
  
  x <- data.table::copy(dat)
  grid <- quantile(x$prediction, prob = seq(0.01, 0.99, 0.01))
  
  x[, is_case := any(label), by = c(id_vars(x))]
  
  onset <- x[label == T, head(.SD, 1L), by = c(id_vars(x))]
  onset <- onset[, c("stay_id", "stay_time"), with=F]
  onset <- rename_cols(onset, "onset_time", "stay_time")
  onset[, onset_time := onset_time + hours(6L)]
  onset <- as_id_tbl(onset)
  
  x <- merge(x, onset, by = id_vars(x), all.x = T)
  
  res <- NULL
  for (i in 1:length(grid)) {
    
    thresh <- grid[i]
    
    trig <- x[prediction > thresh, head(.SD, 1L), by = c(id_vars(x))]
    trig <- trig[, c("stay_id", "stay_time"), with=F]
    trig <- rename_cols(trig, "trigger", "stay_time")
    trig <- as_id_tbl(trig)
    
    fin <- merge(
      unique(x[, c(id_vars(x), "is_case", "onset_time"), with = F]),
      trig, all.x = T
    )
    
    dcs <- as.vector(table(fin$is_case, !is.na(fin$trigger)))
    tn <- dcs[1]
    fn <- dcs[2]
    fp <- dcs[3]
    tp <- dcs[4]
    
    erl <- fin[!is.na(onset_time) & !is.na(trigger), 
               median(onset_time - trigger)]
    
    
    sens <- tp / (tp + fn)
    spec <- tn / (tn + fp)
    ppv <- tp / (tp + fp)
    
    res <- rbind(res, c(i/100, sens, spec, ppv, as.numeric(erl)))
    
  }
  
  res <- as.data.frame(res)
  names(res) <- c("threshold", "sens", "spec", "ppv", "earliness")
  
  roc <- ggplot(res, aes(x = 1-spec, y = sens)) + 
    geom_line(color = "blue") +
    theme_bw() + ylim(c(0, 1)) + xlim(c(0,1)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dotdash") + 
    ggtitle("ROC curve")
  
  prc <- ggplot(res, aes(x = sens, y = ppv)) + geom_line(color = "red") +
    theme_bw() + ylim(c(0, 1)) + ggtitle("PR curve") +
    geom_hline(yintercept = mean(fin$is_case), linetype = "dotdash")
  
  earl <- ggplot(res, aes(x = threshold, y = earliness)) + 
    geom_line(color = "green") +
    theme_bw() + ggtitle("Earliness curve")
  
  cowplot::plot_grid(roc, prc, earl, ncol = 3L)
  
}

#patient_eval(dat)
