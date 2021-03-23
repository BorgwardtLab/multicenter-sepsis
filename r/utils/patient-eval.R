
patient_eval <- function(dat) {

  integrate <- function(x, y) {

    assert_that(length(x) == length(y))

    x_left <- x[1:(length(x)-1)]
    x_right <- x[2:(length(x))]

    y_left <- y[1:(length(x)-1)]
    y_right <- y[2:(length(x))]

    sum((x_right - x_left) * (y_left + y_right) / 2)

  }

  convex_comb <- function(a, b, val1, val2, mid = 0.9) {

    assertthat::assert_that(a >= mid, b <= mid)

    val2 + (val1 - val2) * (mid - b) / (a - b)

  }

  x <- data.table::copy(dat)

  prob_grid <- c(0, 0.001, seq(0.01, 0.99, 0.01), 0.999)
  grid <- quantile(x[, max(prediction), by = "stay_id"][["V1"]],
                   prob = prob_grid)

  x <- x[, is_case := any(!is.na(onset)), by = c(id_vars(x))]
  onset <- x[!is.na(onset), head(.SD, 1L), by = c(id_vars(x)),
             .SD = "onset"]
  onset <- onset[, c("onset_time", "onset") := list(
    as.difftime(onset, units = "hours"), NULL)]

  x <- merge(x, onset, by = id_vars(x), all.x = TRUE)

  util_opt <- sum(x[["utility"]][x[["utility"]] > 0])
  res <- c(1, 0, 1, 1, 0, 0, 0)

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
               list(onset_time - trigger)][["V1"]]

    tn <- dcs[1]
    fn <- dcs[2]
    fp <- dcs[3]
    tp <- dcs[4]

    sens <- tp / (tp + fn)
    spec <- tn / (tn + fp)
    ppv <- tp / (tp + fp)
    util <- sum((x[["prediction"]] > thresh) * x[["utility"]])
    util <- util / util_opt

    res <- rbind(res, c(prob_grid[i], sens, spec, ppv, as.numeric(median(erl)),
                        mean(erl > hours(2L)), util))
  }

  res <- as.data.frame(res)
  names(res) <- c("threshold", "sens", "spec", "ppv", "earliness", "advance_2h",
                  "utility")
  res <- res[order(res$threshold), ]

  prec_90r <- earliness_90r <- advance_90r <- -Inf
  for (i in 1:nrow(res)) {

    if(res$sens[i] > 0.9 & res$sens[i+1] <= 0.9) {

      prec_90r <- convex_comb(res$sens[i], res$sens[i+1], res$ppv[i], res$ppv[i+1])
      earliness_90r <- convex_comb(res$sens[i], res$sens[i+1], res$earliness[i],
                                   res$earliness[i+1])
      advance_90r <- convex_comb(res$sens[i], res$sens[i+1], res$advance_2h[i],
                                 res$advance_2h[i+1])

    }

  }


  auroc <- -integrate(1 - res$spec, res$sens)
  auprc <- -integrate(res$sens, res$ppv)

  structure(
    list(
      dat = res,
      prop_sep = mean(fin$is_case),
      auroc = -integrate(1 - res$spec, res$sens),
      auprc = -integrate(res$sens, res$ppv),
      max_util = max(res$utility),
      prec_90r = prec_90r,
      earliness_90r = earliness_90r,
      advance_90r = advance_90r
    ),
    class = "eval"
  )
}

patient_plot <- function(dat, run = NULL) {

  if (!inherits(dat, "eval")) {
    dat <- patient_eval(dat)
  }

  assert_that(inherits(dat, "eval"))

  roc <- ggplot(dat$dat, aes(x = 1-spec, y = sens)) +
    geom_line(color = "blue") +
    theme_bw() + ylim(c(0, 1)) + xlim(c(0,1)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dotdash") +
    ggtitle(
      paste0("ROC curve of ", run, " with AUROC ", round(dat$auroc, 4))
    )

  prc <- ggplot(dat$dat, aes(x = sens, y = ppv)) + geom_line(color = "red") +
    theme_bw() + ylim(c(0, 1)) +
    geom_hline(yintercept = dat$prop_sep, linetype = "dotdash") +
    ggtitle(
      paste0("AUPRC ", round(dat$auprc, 4), "; At 90% rec. precision ",
             round(dat$prec_90r*100, 2), "%")
    ) +
    geom_vline(xintercept = 0.9, linetype = "dashed", color = "red")

  earl <- ggplot(dat$dat, aes(x = sens, y = earliness)) +
    geom_line(color = "green") + theme_bw() +
    ggtitle(
      paste0("At 90% rec. med. earl. ",
             round(dat$earliness_90r, 2), "; ",
             round(dat$advance_90r*100, 2), "% >=2h before"
            )
    ) +
    geom_vline(xintercept = 0.9, linetype = "dashed", color = "red")

  phys <- ggplot(dat$dat, aes(x = threshold, y = utility)) +
    geom_line(color = "pink") + theme_bw() +
    ggtitle(paste0("Utility curve with maximum ", round(dat$max_util, 4)))

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

  util <- if (length(res$utility)) res$utility else res$labels

  res <- data.frame(
    stay_id = rep(as.integer(res$ids), lengths(res$times)),
    stay_time = do.call(c, res$times),
    prediction = do.call(c, res$scores),
    label = do.call(c, res$labels),
    utility = do.call(c, util),
    onset = rep(as.integer(res$onset), lengths(res$times))
  )

  try_id_tbl(res)
}
