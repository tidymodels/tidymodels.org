library(downlit)
library(xml2)

# Workaround for https://github.com/r-lib/downlit/issues/173
# downlit special-cases library(tidyverse) but not library(tidymodels), so its
# member packages are never added to the search path used for linking.
# Pre-seeding downlit.attached fixes linking for step_*, boost_tree(), tune(), etc.
# Equivalent to tidymodels::tidymodels_packages() but avoids a tidymodels dependency in CI
tidymodels_pkgs <- c(
  "broom", "cli", "conflicted", "dials", "dplyr", "ggplot2", "hardhat",
  "infer", "modeldata", "parsnip", "purrr", "recipes", "rlang", "rsample",
  "rstudioapi", "tailor", "tibble", "tidymodels", "tidyr", "tune",
  "workflows", "workflowsets", "yardstick"
)
options(downlit.attached = union(tidymodels_pkgs, getOption("downlit.attached")))

# Collect HTML output files
output_files <- strsplit(Sys.getenv("QUARTO_PROJECT_OUTPUT_FILES"), "\n")[[1]]
html_files   <- output_files[grepl("\\.html$", output_files)]

# Fallback: scan _site/ directly if env var is empty (e.g. all pages from freeze cache)
if (length(html_files) == 0) {
  message("downlit: QUARTO_PROJECT_OUTPUT_FILES empty, scanning _site/ for HTML files")
  html_files <- list.files("_site", pattern = "\\.html$", recursive = TRUE, full.names = TRUE)
}

if (length(html_files) == 0) {
  message("downlit: no HTML files found, skipping")
  quit(status = 0)
}

message(sprintf("downlit: linking R functions in %d HTML file(s)", length(html_files)))

for (file in html_files) {
  downlit::downlit_html_path(file, file, classes = downlit::classes_pandoc())
}

message("downlit: done")
