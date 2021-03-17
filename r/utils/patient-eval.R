
patient_eval <- function(dat, run) {
  
  integrate <- function(x, y) {
    
    assertthat::assert_that(length(x) == length(y))
    
    x_left <- x[1:(length(x)-1)]
    x_right <- x[2:(length(x))]
    
    y_left <- y[1:(length(x)-1)]
    y_right <- y[2:(length(x))]
    
    sum((x_right - x_left) * (y_left + y_right) / 2)
    
  }

  x <- data.table::copy(dat)

  prob_grid <- c(0, 0.001, seq(0.01, 0.99, 0.01), 0.999)
  grid <- quantile(x$prediction, prob = prob_grid)

  x[, is_case := any(label), by = c(id_vars(x))]

  onset <- x[label == TRUE, head(.SD, 1L), by = c(id_vars(x))]
  onset <- onset[, c("stay_id", "stay_time"), with = FALSE]
  onset <- rename_cols(onset, "onset_time", "stay_time")
  onset[, onset_time := onset_time + hours(6L)]
  onset <- as_id_tbl(onset)

  x <- merge(x, onset, by = id_vars(x), all.x = TRUE)

  util_opt <- sum(x[["utility"]][x[["utility"]] > 0])
  res <- c(1, 0, 1, 1, 0, 0)

  for (i in 1:length(grid)) {

    thresh <- grid[i]
    
    trig <- x[prediction > thresh, head(.SD, 1L), by = c(id_vars(x))]
    trig <- trig[, c("stay_id", "stay_time"), with = FALSE]
    trig <- rename_cols(trig, "trigger", "stay_time")
    trig <- as_id_tbl(trig)

    fin <- merge(
      unique(x[, c(id_vars(x), "is_case", "onset_time"), with = FALSE]),
      trig, all.x = TRUE
    )
    
    

    dcs <- as.vector(
      table(fin$is_case, factor(!is.na(fin$trigger), levels = c(F, T)))
    )
    
    erl <- fin[!is.na(onset_time) & !is.na(trigger),
               median(onset_time - trigger)]

    tn <- dcs[1]
    fn <- dcs[2]
    fp <- dcs[3]
    tp <- dcs[4]

    sens <- tp / (tp + fn)
    spec <- tn / (tn + fp)
    ppv <- tp / (tp + fp)
    util <- sum((x[["prediction"]] > thresh) * x[["utility"]])
    util <- util / util_opt

    res <- rbind(res, c(prob_grid[i], sens, spec, ppv, as.numeric(erl), util))

  }

  res <- as.data.frame(res)
  names(res) <- c("threshold", "sens", "spec", "ppv", "earliness", "utility")
  res <- res[order(res$threshold), ]
  
  auroc <- -integrate(1 - res$spec, res$sens)
  auprc <- -integrate(res$sens, res$ppv)
  roc <- ggplot(res, aes(x = 1-spec, y = sens)) +
    geom_line(color = "blue") +
    theme_bw() + ylim(c(0, 1)) + xlim(c(0,1)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dotdash") +
    ggtitle(paste0("ROC curve of ", run, " with AUROC ", 
                   round(auroc, 4)))

  prc <- ggplot(res, aes(x = sens, y = ppv)) + geom_line(color = "red") +
    theme_bw() + ylim(c(0, 1)) + 
    geom_hline(yintercept = mean(fin$is_case), linetype = "dotdash") +
    ggtitle(paste0("PR curve of ", run,  " with AUPRC ", round(auprc, 4)))

  earl <- ggplot(res, aes(x = threshold, y = earliness)) +
    geom_line(color = "green") +
    theme_bw() + ggtitle("Earliness curve")
  
  phys <- ggplot(res, aes(x = threshold, y = utility)) +
    geom_line(color = "pink") + theme_bw() +
    ggtitle(paste0("Utility curve with maximum ", round(max(res$utility), 4)))
    
  cowplot::plot_grid(roc, prc, earl, phys, ncol = 2L, labels = "auto")

}

read_res <- function(train_src = "mimic_demo", test_src = train_src,
                     feat_set = c("basic", "wav", "sig", "full"),
                     predictor = c("linear", "rf"),
                     target = c("class", "hybrid", "reg"),
                     dir = data_path("res"), jobid = NULL, fix_order = FALSE) {

  if (is.null(jobid)) {

    dir <- paste0("^", file.path(dir, "model_"), "[0-9]+")
    dir <- grep(dir, list.dirs(dir), value =TRUE)
    dir <- tail(dir, n = 1L)

    jobid <- sub("^model_", "", basename(dir))

  } else {

    dir <- file.path(dir, paste0("model_", jobid))
  }

  assert_that(dir.exists(dir))

  fil <- list.files(dir,
    paste(predictor, target, feat_set, train_src, test_src, sep = "-"),
    full.names = TRUE
  )

  if (length(fil) == 0L) {
    return(NULL)
  }

  res <- jsonlite::read_json(fil, simplifyVector = TRUE, flatten = TRUE)

  if (fix_order || as.integer(jobid) == 165674415L) {
    perm <- order(as.integer(res$ids))
    res$ids <- res$ids[perm]
    res$times <- res$times[perm]
    res$onset <- res$onset[perm]
  }

  res <- data.frame(
    stay_id = rep(as.integer(res$ids), lengths(res$times)),
    stay_time = do.call(c, res$times),
    prediction = do.call(c, res$scores),
    label = do.call(c, res$labels),
    utility = do.call(c, res$utility),
    onset = rep(as.integer(res$onset), lengths(res$times))
  )

  try_id_tbl(res)
}
