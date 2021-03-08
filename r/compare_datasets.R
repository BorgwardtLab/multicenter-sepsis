
invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

library(ggplot2)
library(ggnewscale)
library(cowplot)

true_runs <- function(x) {
  tmp <- rle(x)
  tmp$lengths[!tmp$values]
}

demo <- c("mimic_demo", "eicu_demo")
prod <- c("mimic", "eicu", "hirid", "aumc", "physionet2019")

datasets <- prod

stats <- vector("list", length(datasets))
names(stats) <- datasets

for (ds in datasets) {

  dat <- read_parquet(file.path(data_path(), ds),
    c("stay_id", "stay_time", "onset", "ts_miss", "ts_avail")
  )

  stay <- dat[, list(los = max(get(index_var(dat))),
                     is_case = any(!is.na(onset))), by = c(id_var(dat))]
  sep  <- dat[get("onset"), ]
  miss <- dat[stay_time > 0, true_runs(ts_avail), by = stay_id]

  n_stay <- nrow(stay)
  n_sep  <- nrow(sep)

  stats[[ds]] <- list(
    source = ds,
    n_stay = n_stay,
    los = as.double(stay[["los"]], units = "hours"),
    n_sep = n_sep,
    onset = as.double(index_col(sep), units = "hours"),
    miss_rle = miss$V1,
    miss_rate = dat[stay_time > 0, ts_miss / max(ts_miss)],
    case = stay[["is_case"]]
  )
}

stay_hist <- lapply(stats, `[`, c("source", "los"))
stay_hist <- do.call(rbind, lapply(stay_hist, as.data.frame))
stay_hist <- ggplot(stay_hist, aes(source, los)) +
  geom_boxplot()

onset_hist <- lapply(stats, `[`, c("source", "onset"))
onset_hist <- do.call(rbind, lapply(onset_hist, as.data.frame))
onset_hist <- ggplot(onset_hist, aes(source, onset)) +
  geom_boxplot()

lines <- c(0.05, 0.1, 0.2, 0.3)
lines <- data.frame(slope = lines, icept = 0,
  label = ordered(paste0(lines * 100, "%"), levels = paste0(lines * 100, "%"))
)

preva <- lapply(stats, `[`, c("source", "n_stay", "n_sep"))
preva <- do.call(rbind, lapply(preva, as.data.frame))
preva <- ggplot(preva, aes(n_stay, n_sep)) +
  geom_point(aes(color = source)) +
  new_scale_color() +
  geom_abline(data = lines,
    aes(intercept = icept, slope = slope, color = label),
    show.legend = TRUE)

miss <- lapply(stats, `[`, c("source", "miss_rle"))
miss <- do.call(rbind, lapply(miss, as.data.frame))
miss <- ggplot(miss, aes(miss_rle, fill = source)) +
  stat_count() +
  scale_y_log10()

rate <- lapply(stats, `[`, c("source", "miss_rate"))
rate <- do.call(rbind, lapply(rate, as.data.frame))
rate <- ggplot(rate, aes(miss_rate)) +
  geom_density(aes(color = source))

ratio <- lapply(stats, `[`, c("source", "los", "case"))
ratio <- do.call(rbind, lapply(ratio, as.data.frame))
ratio <- ggplot(ratio, aes(los, fill = case)) +
  stat_count() +
  facet_grid(rows = vars(source))

plt <- plot_grid(
  stay_hist, onset_hist, preva, miss, ratio, rate,
  labels = LETTERS[1:6],
  ncol = 3L
)

ggsave("compare_datasets.pdf", plt, width = 18, height = 12)
