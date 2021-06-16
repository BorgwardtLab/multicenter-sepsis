
invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

add_tail <- function(x) {
  rbind(x, tail(x, n = 1L)[, data_vars(x) := NA][, stay_time := mins(1860L)])
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

  min <- load_concepts(feat, ..., interval = mins(1L))
  hrs <- load_concepts(feat, ..., interval = mins(60L))
  hrs <- change_interval(hrs, mins(1L))

  min <- rename_cols(min, "min", data_vars(min))
  hrs <- rename_cols(hrs, "hrs", data_vars(hrs))

  res <- merge(min, hrs, all = TRUE)
  res <- rename_cols(res, "stay_time", index_var(min))
  res <- filter_time(res)
  res <- replace_na(add_tail(res), type = "locf", vars = "hrs")

  stats <- read_colstats(src, cols = paste0(feat, "_raw"))

  res <- res[, c(data_vars(res)) := lapply(.SD, zscore, stats["means"],
    stats["stds"]), .SDcols = data_vars(res)]

  as_hours(res)
}

src <- "aumc"

dat <- read_res(train_src = src, test_src = src, feat_set = "locf",
                predictor = "lgbm", prefix = "cube")

pid <- unique(dat[is_case & onset > hours(24) & onset < hours(26), ]$stay_id)

vitals <- lapply(
  c("resp", "o2sat", "hr"),
  function(feat, ...) {

    res <- load_dat(feat, src, patient_ids = pid)

    res <- ggplot(res) +
      geom_point(aes(stay_time, min), alpha = 0.2) +
      geom_step(aes(stay_time, hrs), color = "red") +
      theme_bw() +
      xlim(-1, 31) +
      ylab(feat)

    if (identical(feat, "hr")) {
      res +
        xlab("Stay time relative to ICU admission [hours]") +
        ylab("Heart rate")
    } else {
      res <- rm_x_axis(res)
      if (identical(feat, "resp")) {
        res + ylab("Resp. rate")
      } else {
        res + ylab(expression(O[2]~Sat.))
      }
    }
  }
)

labs <- lapply(
  c("crea", "bili", "lact"),
  function(feat, ...) {
    res <- load_dat(feat, src, patient_ids = pid)
    res <- res[, feat := feat]
    res
  }
)

labs <- rbind_lst(labs)

labp <- ggplot(labs) +
  geom_point(aes(stay_time, min, color = feat)) +
  geom_step(aes(stay_time, hrs, color = feat)) +
  theme_bw() +
  xlim(-1, 31) +
  ylab("Lab tests") +
  guides(color = guide_legend(title = "Lab test"))

pred <- ggplot(as_hours(filter_time(dat[stay_id == pid, ]))) +
  geom_step(aes(stay_time, prediction)) +
  geom_vline(xintercept = 25, color = "red", linetype = "dashed") +
  theme_bw() +
  xlim(-1, 31) +
  ylab("Prediction score")

sabx <- load_concepts(c("samp", "abx"), src, patient_ids = pid, merge = FALSE,
                      interval = mins(1L))

sabx <- lapply(sabx,
  function(x) {
    x <- x[, c("feat", data_vars(x)) := list(data_vars(x), NULL)]
    rename_cols(x, "stay_time", index_var(x))
  }
)
sabx <- rbind_lst(sabx)
sabx <- rename_cols(sabx, "stay_time", index_var(sabx))

susi <- load_concepts("susp_inf", src, patient_ids = pid, interval = mins(60L))
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

soda <- load_concepts("sofa", src, patient_ids = pid)
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
    plotlist = c(
      list(rm_x_axis(sofa), rm_x_axis(swin), rm_x_axis(pred),
           rm_x_axis(labp) + theme(legend.position = "none")),
      vitals
    ),
    ncol = 1L,
    rel_heights = c(1, 0.5, 1.5, 1, 1, 1, 1.25),
    align = "v"
  ),
  cowplot::get_legend(labp + theme(legend.position = "bottom")),
  ncol = 1L,
  rel_heights = c(2, 0.1)
)

ggsave("prediction.pdf", width = 7, height = 10.5)
