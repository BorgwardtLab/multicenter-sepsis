library(ricu)
library(ggplot2)

fpos_Module <- function(delta, L = -12, R = 4, mid = -6, U_fp = -0.05) {
  
  if (inherits(delta, "difftime")) delta <- as.double(delta, units = "hours")
  
  data.table::fifelse(
    delta < L, U_fp,
    data.table::fifelse(
      delta <= mid, (delta - L) / (mid - L),
      data.table::fifelse(
        delta <= R, 1 - (delta - mid) / (R - mid), 0
      )
    )
  )
  
}

rng <- -15:6
fpos_Module(rng)

mid <- seq.int(-8, 2, 2)

df <- as.data.frame(
  Reduce(rbind, lapply(mid, function(m, rng) {
    
    cbind(rng, fpos_Module(rng, mid = m), m)
    
  }, rng = rng))
)

ggplot(df, aes(x = rng, y = V2, color = factor(m))) +
  geom_line() + ylab("f positive module") + xlab("Delta(t)")

# range to explore (suggestion)

ufp_grid <- c(-0.01, 0.025, -0.05, -0.1, -0.2, -0.5, -1)
mid <- -8:3

expand.grid(ufp_grid, mid)

