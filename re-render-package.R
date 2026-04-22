# ------------------------------------------------------------------------------
# Selectively re-render pages that depend on one or more R packages.
# Run: Rscript re-render-package.R <pkg1> [pkg2 ...]
#
# Examples:
#   Rscript re-render-package.R ranger
#   Rscript re-render-package.R ranger glmnet
#   Rscript re-render-package.R tidymodels   # re-renders all tidymodels pages

library(jsonlite)
library(cli)

# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------
# Resolve pages for all requested packages (union, deduped)

unknown <- setdiff(args, names(pkg_map))
if (length(unknown) > 0) {
  cli::cli_warn("No pages found for package{?s}: {.pkg {unknown}}")
}

known <- intersect(args, names(pkg_map))
if (length(known) == 0) {
  cli::cli_abort("None of the requested packages appear in {.file package_map.json}.")
}

pages <- sort(unique(unlist(pkg_map[known], use.names = FALSE)))

cli::cli_h1("Selective re-render")
cli::cli_alert_info("Package{?s}: {.pkg {args}}")
cli::cli_alert_info("{length(pages)} page{?s} to re-render")

# ------------------------------------------------------------------------------
# Clear freeze cache for affected pages, then render

for (page in pages) {
  rel_dir    <- dirname(page)
  freeze_dir <- file.path(repo_root, "_freeze", rel_dir)

  if (dir.exists(freeze_dir)) {
    unlink(freeze_dir, recursive = TRUE)
    cli::cli_alert("Cleared freeze cache: {.path {freeze_dir}}")
  }

  cli::cli_h2("Rendering {.file {page}}")
  result <- system2("quarto", c("render", file.path(repo_root, page)))

  if (result != 0) {
    cli::cli_warn("Render failed for {.file {page}}")
  }
}

cli::cli_alert_success("Done. {length(pages)} page{?s} re-rendered.")
