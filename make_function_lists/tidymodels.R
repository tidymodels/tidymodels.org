# ------------------------------------------------------------------------------
# Generate tidymodels function reference list
# Run: Rscript make_function_lists/tidymodels.R

source(here::here("make_function_lists/_utils.R"))

# ------------------------------------------------------------------------------

all_tm <- read_csv(here::here("make_function_lists/all_packages.csv"), show_col_types = FALSE)$name

# Cache available packages to avoid repeated CRAN metadata fetches
avail <- get_available_packages()

tidymodels_functions <-
  purrr::map_dfr(
    all_tm,
    get_pkg_info,
    available = avail,
    .progress = TRUE
  ) %>%
  sort_out_urls() %>%
  dplyr::filter(!grepl("^\\.", functions)) %>%
  dplyr::rename(func = functions) %>%
  dplyr::select(title, func, url, package)

tidymodels_functions[is.na(tidymodels_functions)] <- ""

items <- lapply(seq_len(nrow(tidymodels_functions)), function(i) {
  list(
    title = tidymodels_functions$title[i],
    func = tidymodels_functions$func[i],
    package = as.character(tidymodels_functions$package[i]),
    url = tidymodels_functions$url[i]
  )
})

yaml::write_yaml(items, here::here("find/all/items.yml"))

cli::cli_alert_success("Generated find/all/items.yml")
