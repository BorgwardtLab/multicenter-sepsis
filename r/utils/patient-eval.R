
patient_eval <- function(dat, split_cols = NULL, earl_loc = median,
                         prob_grid = c(0.001, seq(0.01, 0.99, 0.01), 0.999),
                         unique_cols = split_cols) {

  integrate <- function(x, y) {

    assert_that(length(x) == length(y))

    len1 <- seq_len(length(x) - 1)
    len2 <- seq.int(2L, length(x))

    sum((x[len2] - x[len1]) * (y[len1] + y[len2]) / 2)
  }

  convex_comb <- function(a, b, val1, val2, mid = 0.9) {

    assert_that(a >= mid, b <= mid)

    val2 + (val1 - val2) * (mid - b) / (a - b)
  }

  if (!is.null(split_cols)) {

    dat <- split(dat, by = split_cols)
    res <- lapply(dat, patient_eval, earl_loc = earl_loc,
                  prob_grid = prob_grid, unique_cols = unique_cols)

    atr <- data.table::rbindlist(lapply(res, attr, "stats"))
    res <- data.table::rbindlist(res)

    res <- data.table::setattr(res, "stats", atr)
    res <- data.table::setattr(res, "class", c("patient_eval", class(res)))

    return(res)
  }

  meta <- unique(dat[,
    unique(c("train_src", "test_src", "feat_set", "predictor",
             "target", "split", unique_cols)), with = FALSE
  ])

  assert_that(nrow(meta) == 1L)

  grid <- dat[, list(max_pred = max(prediction)), by = "stay_id"]
  grid <- quantile(grid[["max_pred"]], prob = prob_grid)

  opt_util <- dat[["utility"]]
  opt_util <- sum(opt_util[opt_util > 0])

  res <- matrix(nrow = length(grid), ncol = 7L,
    dimnames = list(NULL, c("threshold", "sens", "spec", "ppv", "earliness",
                            "advance_2h", "utility"))
  )

  for (i in seq_along(grid)) {

    thresh <- grid[i]

    trig <- dat[prediction > thresh, head(.SD, 1L), by = "stay_id",
                .SDcols = "stay_time"]
    trig <- rename_cols(trig, "trigger", "stay_time")

    fin <- merge(
      unique(dat[, c("stay_id", "is_case", "onset"), with = FALSE]),
      trig, all.x = TRUE
    )

    dcs <- table(fin[["is_case"]], !is.na(fin[["trigger"]]))

    erl <- fin[!is.na(onset) & !is.na(trigger),
               list(ealiness = onset - trigger)]
    erl <- as.double(erl[["ealiness"]], units = "hours")

    tn <- dcs["FALSE", "FALSE"]
    fn <- dcs["TRUE", "FALSE"]
    fp <- dcs["FALSE", "TRUE"]
    tp <- dcs["TRUE", "TRUE"]

    sens <- tp / (tp + fn)
    spec <- tn / (tn + fp)
    ppv  <- tp / (tp + fp)

    util <- sum((dat[["prediction"]] > thresh) * dat[["utility"]])
    util <- util / opt_util

    res[i, ] <- c(
      prob_grid[i], sens, spec, ppv, earl_loc(erl), mean(erl > 2), util
    )
  }

  res <- data.table::as.data.table(res)
  res <- cbind(res, meta)

  prec_90r <- earliness_90r <- advance_90r <- -Inf

  for (i in seq_len(nrow(res))) {

    i1 <- i + 1

    if (res$sens[i] > 0.9 & res$sens[i1] <= 0.9) {

      prec_90r <- convex_comb(
        res$sens[i], res$sens[i1], res$ppv[i], res$ppv[i1]
      )

      earliness_90r <- convex_comb(
        res$sens[i], res$sens[i1], res$earliness[i], res$earliness[i1]
      )

      advance_90r <- convex_comb(
        res$sens[i], res$sens[i1], res$advance_2h[i], res$advance_2h[i1]
      )
    }
  }

  prop_sep <- dat[, list(is_case = is_case[1L]), by = "stay_id"]
  prop_sep <- mean(prop_sep[["is_case"]])

  stats <- data.table::data.table(
    prop_sep = prop_sep,
    auroc = -integrate(1 - res$spec, res$sens),
    auprc = -integrate(res$sens, res$ppv),
    max_util = max(res$utility),
    prec_90r = prec_90r,
    earliness_90r = earliness_90r,
    advance_90r = advance_90r
  )

  res <- data.table::setattr(res, "stats", cbind(stats, meta))
  res <- data.table::setattr(res, "class", c("patient_eval", class(res)))

  res
}

patient_plot <- function(dat, ..., mod_col = "predictor", rep_col = NULL) {

  interpolate <- function(x, ...) {
    x <- x[, suppressWarnings(
      approx(x, y, seq(0, 1, length.out = 1e3), ...)
    ), by = c("mod", "rep")]
    x[!is.na(y), list(y_min = min(y), y_med = median(y), y_max = max(y)),
      by = c("mod", "x")]
  }

  plot_ribbon <- function(x, has_reps) {

    if (has_reps) {

      res <- ggplot(x, aes(x = x, y = y_med)) +
        geom_ribbon(aes(ymin = y_min, ymax = y_max, group = mod), alpha = 0.25,
                    fill = "grey25", na.rm = TRUE)
    } else {

      res <- ggplot(x, aes(x = x, y = y))
    }

    res + geom_smooth(aes(color = mod), stat = "identity", size = 0.5,
                      na.rm = TRUE) +
      theme_bw()
  }

  summarize <- function(x, has_reps) {
    x <- signif(x, 3)
    if (has_reps) {
      paste0(median(x), " (", min(x), " - ", max(x), ")")
    } else {
      paste(x)
    }
  }

  sum2 <- function(x) {
    paste(c("AUROC", "AUPRC", "Prec. @90R", "Earl. @90R"), x, sep = ": ",
          collapse = ", ")
  }

  if (!inherits(dat, "patient_eval")) {
    dat <- patient_eval(dat, ...)
  }

  assert_that(inherits(dat, "patient_eval"))

  meta <- attr(dat, "stats")

  assert_that(length(unique(meta$test_src)) == 1L)

  if (is.null(rep_col)) {

    roc <- dat[, list(x = 1 - spec, y = sens, mod = get(mod_col))]
    prc <- dat[, list(x = sens, y = ppv, mod = get(mod_col))]
    phy <- dat[, list(x = threshold, y = utility, mod = get(mod_col))]
    erl <- dat[, list(x = sens, y = earliness, mod = get(mod_col))]

  } else {

    roc <- interpolate(
      dat[, list(x = 1 - spec, y = sens, mod = get(mod_col),
                 rep = get(rep_col))],
      yleft = 0, yright = 1
    )

    prc <- interpolate(
      dat[sens > 0, list(x = sens, y = ppv, mod = get(mod_col),
                         rep = get(rep_col))],
      yleft = 1, yright = meta[["prop_sep"]][1L]
    )

    phy <- interpolate(
      dat[, list(x = threshold, y = utility, mod = get(mod_col),
                 rep = get(rep_col))]
    )

    erl <- interpolate(
      dat[, list(x = sens, y = earliness, mod = get(mod_col),
                 rep = get(rep_col))]
    )
  }

  roc <- plot_ribbon(roc, !is.null(rep_col)) +
    ylim(c(0, 1)) + xlim(c(0, 1)) +
    ylab("Sensitivity") + xlab("1 - Specificity") +
    geom_abline(slope = 1, intercept = 0, colour = "grey", linetype = 3)

  prc <- plot_ribbon(prc, !is.null(rep_col)) +
    ylim(c(0, 1)) + xlim(c(0, 1)) +
    ylab("Precision") + xlab("Recall") +
    geom_hline(yintercept = meta[["prop_sep"]][1L], colour = "grey",
               linetype = 3)

  phy <- plot_ribbon(phy, !is.null(rep_col)) +
    ylab("Utility") + xlab("Threshold")


  erl <- plot_ribbon(erl, !is.null(rep_col)) +
    xlim(c(0, 1)) +
    ylab("Earliness") + xlab("Recall") +
    geom_vline(xintercept = 0.9, colour = "grey", linetype = 3)

  meta <- meta[, list(stats = sum2(lapply(.SD, summarize, !is.null(rep_col)))),
    by = mod_col,
    .SDcols = c("auroc", "auprc", "prec_90r", "earliness_90r")
  ]

  leg <- paste0(meta[[mod_col]], ": ", meta[["stats"]])
  leg <- cowplot::get_legend(
    roc +
      scale_colour_discrete("Model", breaks = meta[[mod_col]], labels = leg) +
      guides(color = guide_legend(ncol = 1L)) +
      theme(legend.position = "bottom")
  )

  res <- cowplot::plot_grid(
    roc + theme(legend.position = "none"),
    prc + theme(legend.position = "none"),
    phy + theme(legend.position = "none"),
    erl + theme(legend.position = "none"),
    ncol = 2L, labels = "auto"
  )

  cowplot::plot_grid(res, leg, ncol = 1, rel_heights = c(2, nrow(meta) * 0.1))
}

read_res <- function(train_src = "mimic_demo", test_src = train_src,
                     feat_set = c("basic", "wav", "sig", "full"),
                     predictor = c("linear", "rf"),
                     target = c("class", "hybrid", "reg"), split = "split_0",
                     dir = data_path("res"), jobid = NULL) {

  if (is.null(jobid)) {

    dir <- paste0("^", file.path(dir, "model_"), "[0-9]+")
    dir <- grep(dir, list.dirs(dir), value =TRUE)

  } else {

    dir <- list.files(dir, jobid, full.names = TRUE, include.dirs = TRUE)
    dir <- dir[vapply(dir, is.dir, logical(1L))]
  }

  dir <- tail(dir, n = 1L)

  assert_that(dir.exists(dir))

  fil <- list.files(dir,
    paste(predictor, target, feat_set, train_src, split, test_src, sep = "-"),
    full.names = TRUE
  )

  if (length(fil) == 0L) {
    return(NULL)
  }

  res <- jsonlite::read_json(fil, simplifyVector = TRUE, flatten = TRUE)

  util <- if (length(res$utility)) res$utility else res$labels

  data.table::data.table(
    stay_id = rep(as.integer(res$ids), lengths(res$times)),
    stay_time = hours(do.call(c, res$times)),
    prediction = do.call(c, res$scores),
    label = do.call(c, res$labels),
    utility = do.call(c, util),
    onset = hours(rep(as.integer(res$onset), lengths(res$times))),
    is_case = rep(!is.na(res$onset), lengths(res$times)),
    train_src = train_src,
    test_src = test_src,
    feat_set = feat_set,
    predictor = predictor,
    target = target,
    split = split
  )
}

read_ress <- function(...) {

  call_do <- function(args, what) do.call(what, args)

  combos <- expand.grid(..., KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  colnames(combos) <- names(list(...))
  combos <- split(combos, seq_len(nrow(combos)))

  data.table::rbindlist(lapply(combos, call_do, read_res))
}
