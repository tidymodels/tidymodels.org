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
  dplyr::rename(func = functions) %>%
  dplyr::select(title, func, url, package)

broom_functions[is.na(broom_functions)] <- ""

items <- lapply(seq_len(nrow(broom_functions)), function(i) {
  list(
    title = broom_functions$title[i],
    func = broom_functions$func[i],
    package = as.character(broom_functions$package[i]),
    url = broom_functions$url[i]
  )
})

yaml::write_yaml(items, here::here("find/broom/items.yml"))

cli::cli_alert_success("Generated find/broom/items.yml")
