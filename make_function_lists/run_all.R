# ------------------------------------------------------------------------------
# Run all function list generators
# Run: Rscript make_function_lists/run_all.R
#      Rscript make_function_lists/run_all.R --fresh  (clear cache first)
#
# Or run individual generators:
#   Rscript make_function_lists/parsnip.R
#   Rscript make_function_lists/broom.R
#   Rscript make_function_lists/recipes.R
#   Rscript make_function_lists/tidymodels.R
#   Rscript make_function_lists/sparse.R      (requires parsnip.R and recipes.R)
#   Rscript make_function_lists/tidyclust.R

# Check for --fresh flag to clear cache
args <- commandArgs(trailingOnly = TRUE)
if ("--fresh" %in% args) {
  source(here::here("make_function_lists/_utils.R"))
  clear_pkg_cache()
  cli::cli_alert_info("Cache cleared")
}

cli::cli_h1("Generating function reference lists")

cli::cli_h2("Parsnip models")
source(here::here("make_function_lists/parsnip.R"))

cli::cli_h2("Broom functions")
source(here::here("make_function_lists/broom.R"))

cli::cli_h2("Recipe functions")
source(here::here("make_function_lists/recipes.R"))

cli::cli_h2("Tidymodels functions")
source(here::here("make_function_lists/tidymodels.R"))

cli::cli_h2("Sparse support")
source(here::here("make_function_lists/sparse.R"))

cli::cli_h2("Tidyclust models")
source(here::here("make_function_lists/tidyclust.R"))

cli::cli_h1("Done!")
