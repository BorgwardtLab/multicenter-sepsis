# Step 1
get_param <- function(target, idx, wch = "first") {
  
  params <- list(
    class = list(
      first = seq.int(-12, -2),
      second = seq.int(-2, 8)
    ),
    reg = list(
      first = seq.int(-12, 6, by = 2),
      second = c(0, -0.001, -0.01, -0.02, -0.05, -0.1, -0.2, -0.5, -1, -2)
    )
  )
  
  unlist(Map(function(x, y) params[[x]][[wch]][as.integer(y)], 
                       target, idx))
  
}

# Step 2
res <- data.table::copy(attr(evl, "stats"))
res[, param_1 := get_param(target, targ_param_1, "first")]
res[, param_2 := get_param(target, targ_param_2, "second")]



fix_targ <- c("class", "reg")

p1 <- ggplot(res[target %in% fix_targ], 
       aes(x = earliness_90r, y = prec_90r, color = factor(param_1),
           shape = factor(target))) +
  geom_point() + ylab("Precision at 90% recall") + 
  xlab("Median earliness at 90% recall")

p2 <- ggplot(res[target %in% fix_targ], 
       aes(x = advance_90r, y = prec_90r, color = factor(param_1),
           linetype = factor(target))) +
  geom_line() + ylab("Precision at 90% recall") + 
  xlab("Proportion called >= 2h before")

p3 <- ggplot(res[target %in% fix_targ], 
       aes(x = earliness_90r, y = advance_90r, color = factor(param_1))) +
  geom_point() + xlab("Median earliness at 90% recall") + 
  ylab("Proportion called >= 2h before")

cowplot::plot_grid(p1, p2, p3, ncol = 3L)
