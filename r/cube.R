#!/usr/bin/env Rscript

#BSUB -W 12:00
#BSUB -n 4
#BSUB -R rusage[mem=4000]
#BSUB -J model
#BSUB -o data-res/model_%J.out

invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

src <- "mimic_demo"

feat_combos <- c(
  "_(hours|locf|derived)$",
  "_(hours|locf|derived|wavelet)$",
  "_(hours|locf|derived|signature)$",
  "_(hours|locf|derived|wavelet|signature)$"
)

for (feats in feat_combos) {
  for (pred in c("linear", "rf")) {
    for (targ in c("class", "hybrid", "reg")) {

      msg("running: `", pred, "` model with `", targ, "` response and using `",
          feats, "` selected features")

      prof(
        fit_predict(src, feat_reg = feats, predictor = pred, target = targ)
      )
    }
  }
}
