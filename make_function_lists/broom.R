# ------------------------------------------------------------------------------
# Generate broom function reference list
# Run: Rscript make_function_lists/broom.R

source(here::here("make_function_lists/_utils.R"))

# ------------------------------------------------------------------------------

broom_pkgs <- revdepcheck::cran_revdeps(
  "broom",
  dependencies = c("Depends", "Imports")
)
generics_pkgs <- revdepcheck::cran_revdeps("generics", dependencies = "Imports")

broom_pkgs <- sort(unique(c(broom_pkgs, generics_pkgs)))
excl <- c("hydrorecipes", "healthcareai", "doBy", "nestedmodels", "skedastic")
broom_pkgs <- broom_pkgs[!(broom_pkgs %in% excl)]

# Cache available packages to avoid repeated CRAN metadata fetches
avail <- get_available_packages()

broom_functions <-
  purrr::map_dfr(
    broom_pkgs,
    get_pkg_info,
    pattern = "(^tidy\\.)|(^glance\\.)|(^augment\\.)",
    available = avail,
    .progress = TRUE
  ) %>%
  sort_out_urls() %>%
  dplyr::select(-functions)

write_csv(
  broom_functions,
  file = here::here("find/broom/broom_functions.csv")
)

cli::cli_alert_success("Generated find/broom/broom_functions.csv")
