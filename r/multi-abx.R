
invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

for (src in c("mimic", "aumc")) {

  so <- load_concepts("sofa", src)
  si <- load_concepts("susp_inf", src, abx_min_count = 2L, si_mode = "abx")

  res <- sep3(so, si, si_window = "any")
  res <- res[is_true(get("sep3")), ]
  res <- split(as.double(index_col(res)), id_col(res))

  jsonlite::write_json(res, paste0("multi-abx_", src, ".json"))
}
