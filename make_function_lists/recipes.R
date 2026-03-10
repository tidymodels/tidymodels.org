# ------------------------------------------------------------------------------
# Generate recipes step function reference list
# Run: Rscript make_function_lists/recipes.R

source(here::here("make_function_lists/_utils.R"))

# ------------------------------------------------------------------------------

recipe_pkgs <- revdepcheck::cran_revdeps(
  "recipes",
  dependencies = c("Depends", "Imports")
)
recipe_pkgs <- c(recipe_pkgs, "recipes")

recipe_pkgs <- sort(unique(c(recipe_pkgs)))
excl <- c(
  "hydrorecipes",
  "healthcareai",
  "D2MCS",
  "nestedmodels",
  "tidytof",
  "viraldomain",
  "viralmodels"
)
recipe_pkgs <- recipe_pkgs[!(recipe_pkgs %in% excl)]

# Cache available packages to avoid repeated CRAN metadata fetches
avail <- get_available_packages()

recipe_functions <-
  purrr::map_dfr(
    recipe_pkgs,
    get_pkg_info,
    pattern = "^step_",
    available = avail,
    .progress = TRUE
  ) %>%
  sort_out_urls() %>%
  dplyr::select(-functions)

write_csv(
  recipe_functions,
  file = here::here("find/recipes/recipe_functions.csv")
)

cli::cli_alert_success("Generated find/recipes/recipe_functions.csv")
