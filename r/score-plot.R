
invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

add_tail <- function(x) {
  rbind(x, tail(x, n = 1L)[, data_vars(x) := NA][, stay_time := hours(31L)])
}

filter_time <- function(x) {
  x[stay_time >= hours(-12) & stay_time < hours(31), ]
}

as_hours <- function(x) {
  x$stay_time <- as.double(x$stay_time, units = "hours")
  x
}

rm_x_axis <- function(x) {
  x + theme(axis.title.x = element_blank(),
            axis.text.x = element_blank(),
            axis.ticks.x = element_blank())
}

load_dat <- function(feat, ...) {

  res <- load_concepts(feat, ...)
  res <- rename_cols(res, c("stay_time", "meas"),
                     c(index_var(res), data_var(res)))
  res <- filter_time(res)
  res <- replace_na(add_tail(res), type = "locf", vars = "meas")

  stats <- read_colstats(src, cols = paste0(feat, "_raw"))

  res <- res[, c(data_vars(res)) := lapply(.SD, zscore, stats["means"],
    stats["stds"]), .SDcols = data_vars(res)]

  as_hours(res)
}

src <- "aumc"

# dat <- read_res(train_src = src, test_src = src, feat_set = "locf",
#                 predictor = "lgbm", prefix = "cube")

# dat <- jsonlite::read_json(file.path(data_path("res"),
#                                      "predictions_5330.json"))
# dat <- lapply(dat, function(x) {
#   x <- lapply(x, unlist)
#   l <- lengths(x)
#   x[l == 1] <- lapply(x[l == 1], rep, max(lengths(x)))
#   data.table::as.data.table(x)
# })
# dat <- data.table::rbindlist(dat)
# dat <- dat[times >= -12 & times < 31 & dataset_train == "EICU",
#            c("times", "prob_scores")]

dat <- data.table::data.table(
  times = c(
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30
  ),
  prob_scores = c(
    0.0304742439423084, 0.315896359599296, 0.517189148546229,
    0.629516052045979, 0.698422953960539, 0.710013185986267, 0.780179716033179,
    0.809432176529158, 0.811111307120055, 0.81271707242609, 0.835248976864155,
    0.882805128592015, 0.879238829671948, 0.85299284571392, 0.88677290639147,
    0.907706156332606, 0.903959547086539, 0.907085364827868, 0.90876284042415,
    0.915558170456778, 0.880298371259317, 0.895552229785397, 0.913188222556421,
    0.91182803257856, 0.913253313547549, 0.915294009000581, 0.917794834098309,
    0.914397248571808, 0.911674508092618, 0.910892229283402, 0.912524944263206
  )
)

pid <- unique(dat[is_case & onset > hours(24) & onset < hours(26), ]$stay_id)

vitals <- lapply(
  c("urine", "map", "hr"),
  function(feat, ...) {
    res <- load_dat(feat, ...)
    fea <- switch(feat, urine = "urine", map = "mean arterial bp",
                  hr = "heart rate")
    res <- res[, feat := fea]
    res

  },
  src, patient_ids = pid, verbose = FALSE
)

vitals <- rbind_lst(vitals)

vits <- ggplot(vitals) +
  geom_step(aes(stay_time, meas, color = feat)) +
  theme_bw() +
  xlim(-1, 31) +
  ylab("Vital signs (Z-scored)") +
  xlab("Stay time relative to ICU admission [hours]") +
  guides(color = guide_legend(title = "Vital sign")) +
  theme(legend.position = "bottom")

labs <- lapply(
  c("crea", "bili", "lact"),
  function(feat, ...) {
    res <- load_dat(feat, ...)
    fea <- switch(feat, crea = "creatinine", bili = "bilirubin",
                  lact = "lactate")
    res <- res[, feat := fea]
    res
  },
  src, patient_ids = pid, verbose = FALSE
)

labs <- rbind_lst(labs)

labp <- ggplot(labs) +
  geom_step(aes(stay_time, meas, color = feat)) +
  theme_bw() +
  xlim(-1, 31) +
  ylab("Lab tests (Z-scored)") +
  guides(color = guide_legend(title = "Lab test")) +
  theme(legend.position = "bottom")

pred <- ggplot(dat) +
  geom_step(aes(times, prob_scores)) +
  geom_vline(xintercept = 25, color = "red", linetype = "dashed") +
  geom_hline(yintercept = 0.6381, linetype = "dashed") +
  theme_bw() +
  xlim(-1, 31) +
  ylab("Predicted probability")

sabx <- load_concepts(c("samp", "abx"), src, patient_ids = pid, merge = FALSE,
                      interval = mins(1L), verbose = FALSE)

sabx <- lapply(sabx,
  function(x) {
    x <- x[, c("feat", data_vars(x)) := list(data_vars(x), NULL)]
    rename_cols(x, "stay_time", index_var(x))
  }
)
sabx <- rbind_lst(sabx)
sabx <- rename_cols(sabx, "stay_time", index_var(sabx))

susi <- load_concepts("susp_inf", src, patient_ids = pid, interval = mins(60L),
                      verbose = FALSE)
susi <- change_interval(susi, mins(1L))

susi <- susi[, c("feat", data_vars(susi), "upr", "lwr") := list(
  data_vars(susi),
  NULL,
  as.double(get(index_var(susi)) + hours(24L), units = "hours"),
  as.double(get(index_var(susi)) - hours(48L), units = "hours")
)]

susi <- rename_cols(susi, "stay_time", index_var(susi))

swin <- rbind(sabx, susi[2L, ], fill = TRUE)
swin <- ggplot(as_hours(filter_time(swin)), aes(stay_time, feat)) +
  geom_point(aes(shape = feat)) +
  geom_errorbarh(aes(xmin = lwr, xmax = upr, height = 0.5)) +
  geom_vline(xintercept = 25, color = "red", linetype = "dashed") +
  theme_bw() +
  coord_cartesian(xlim = c(-1, 31)) +
  scale_shape_manual(values = c(4, 4, 32)) +
  theme(legend.position = "none") +
  scale_y_discrete(limits = c("susp_inf", "samp", "abx"),
                   labels = c("SI window", "Sampling", "ABx")) +
  theme(axis.title.y = element_blank())

soda <- load_concepts("sofa", src, patient_ids = pid, verbose = FALSE,
                      keep_components = TRUE)
soda <- rename_cols(soda, "stay_time", index_var(soda))
soda <- soda[, delta_sofa := delta_cummin(sofa)]

sofa <- ggplot(as_hours(filter_time(soda))) +
  geom_step(aes(stay_time, delta_sofa)) +
  geom_hline(yintercept = 2, color = "red", linetype = "dashed") +
  theme_bw() +
  xlim(-1, 31) +
  ylab(expression(Delta~"SOFA"))

cowplot::plot_grid(
  cowplot::plot_grid(
    plotlist = list(rm_x_axis(sofa), rm_x_axis(swin), rm_x_axis(pred),
                    rm_x_axis(labp) + theme(legend.position = "none"),
                    vits + theme(legend.position = "none")),
    ncol = 1L,
    rel_heights = c(1, 0.5, 1, 1, 1.2),
    align = "v"
  ),
  cowplot::get_legend(labp + theme(legend.position = "bottom")),
  cowplot::get_legend(vits + theme(legend.position = "bottom")),
  ncol = 1L,
  rel_heights = c(3, 0.1, 0.1)
)

ggsave("prediction.pdf", width = 7, height = 10.5)
