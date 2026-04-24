# Selectively re-render pages that depend on one or more R packages.
# Run: Rscript re-render-package.R <pkg1> [pkg2 ...]
#
# Examples:
#   Rscript re-render-package.R ranger
#   Rscript re-render-package.R ranger glmnet
#   Rscript re-render-package.R tidymodels   # re-renders all tidymodels pages
#   Rscript re-render-package.R --all        # re-renders every page

library(jsonlite)
source(here::here("R/render_helpers.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  cli::cli_abort("Usage: Rscript re-render-package.R <pkg1> [pkg2 ...]")
}

repo_root <- here::here()
map_path  <- file.path(repo_root, "package_map.json")

if (!file.exists(map_path)) {
  cli::cli_abort(
    "{.file package_map.json} not found. Run {.code Rscript make_package_map.R} first."
  )
}

pkg_map <- jsonlite::read_json(map_path)

if ("--all" %in% args) {
  cli::cli_alert_info("--all flag set: re-rendering all pages")
  pages <- sort(unique(unlist(pkg_map, use.names = FALSE)))
} else {
  unknown <- setdiff(args, names(pkg_map))
  if (length(unknown) > 0) {
    cli::cli_warn("No pages found for package{?s}: {.pkg {unknown}}")
  }
  known <- intersect(args, names(pkg_map))
  if (length(known) == 0) {
    cli::cli_abort("None of the requested packages appear in {.file package_map.json}.")
  }
  pages <- sort(unique(unlist(pkg_map[known], use.names = FALSE)))
}

cli::cli_h1("Selective re-render")
cli::cli_alert_info("Package{?s}: {.pkg {args}}")
render_pages(pages)
