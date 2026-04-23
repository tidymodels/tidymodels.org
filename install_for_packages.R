# ------------------------------------------------------------------------------
# Install only the R packages needed for pages affected by the given packages.
# Run: Rscript install_for_packages.R <pkg1> [pkg2 ...]
#      Rscript install_for_packages.R --all   # install packages for all pages

library(jsonlite)
library(cli)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) == 0) {
  cli::cli_abort("Usage: Rscript install_for_packages.R <pkg1> [pkg2 ...]")
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
# Find affected pages

if ("--all" %in% args) {
  cli::cli_alert_info("--all flag set: installing packages for all pages")
  pages <- sort(unique(unlist(pkg_map, use.names = FALSE)))
} else {
  known   <- intersect(args, names(pkg_map))
  unknown <- setdiff(args, names(pkg_map))

  if (length(unknown) > 0) {
    cli::cli_warn("No pages found for package{?s}: {.pkg {unknown}}")
  }
  if (length(known) == 0) {
    cli::cli_abort("None of the requested packages appear in {.file package_map.json}.")
  }

  pages <- sort(unique(unlist(pkg_map[known], use.names = FALSE)))
}
cli::cli_alert_info("{length(pages)} page{?s} affected")

# ------------------------------------------------------------------------------
# Build reverse map: page -> packages

page_pkgs <- list()
for (pkg in names(pkg_map)) {
  for (page in unlist(pkg_map[[pkg]])) {
    page_pkgs[[page]] <- c(page_pkgs[[page]], pkg)
  }
}

# Collect all packages needed by affected pages
needed <- sort(unique(unlist(page_pkgs[pages], use.names = FALSE)))
cli::cli_alert_info("{length(needed)} package{?s} required: {.pkg {needed}}")

# ------------------------------------------------------------------------------
# Install — handle special cases

# catboost must be installed from GitHub
catboost_ref <- "catboost/catboost/catboost/R-package"
to_install <- ifelse(needed == "catboost", catboost_ref, needed)

pak::pak(to_install, upgrade = TRUE)

# torch must be installed explicitly for brulee to work
if ("brulee" %in% needed) {
  cli::cli_alert_info("Installing torch backend for brulee...")
  torch::install_torch()
}

# sparklyr requires a local Spark installation
if ("sparklyr" %in% needed) {
  cli::cli_alert_info("Installing Spark for sparklyr...")
  sparklyr::spark_install(version = "4.0")
}

cli::cli_alert_success("Done.")
