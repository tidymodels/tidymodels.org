# ------------------------------------------------------------------------------
# Generate tidymodels function reference list
# Run: Rscript make_function_lists/tidymodels.R

source(here::here("make_function_lists/_utils.R"))

# ------------------------------------------------------------------------------

all_tm <- read_csv(here::here("all_packages.csv"), show_col_types = FALSE)$name

tidymodels_functions <-
  purrr::map_dfr(
    all_tm,
    get_pkg_info,
    .progress = TRUE
  ) %>%
  sort_out_urls() %>%
  dplyr::filter(!grepl("^\\.", functions)) %>%
  dplyr::select(-functions)

write_csv(
  tidymodels_functions,
  file = here::here("find/all/tidymodels_functions.csv")
)

cli::cli_alert_success("Generated find/all/tidymodels_functions.csv")
