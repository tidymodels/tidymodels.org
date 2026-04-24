# ------------------------------------------------------------------------------
# Generate a mapping of R packages to the .qmd pages that depend on them.
# Run: Rscript make_package_map.R
#
# Output: package_map.json
#   { "ranger": ["start/resampling/index.qmd", ...], ... }
#
# Pages listing "tidymodels" are expanded to include all tidymodels member
# packages as implicit dependencies.

library(yaml)
library(jsonlite)
library(cli)

# ------------------------------------------------------------------------------
# Helpers

parse_r_packages <- function(path) {
  lines <- readLines(path, warn = FALSE)

  # Find YAML front matter delimiters
  delims <- which(lines == "---")
  if (length(delims) < 2) return(character(0))

  front_matter <- paste(lines[(delims[1] + 1):(delims[2] - 1)], collapse = "\n")
  meta <- tryCatch(yaml::yaml.load(front_matter), error = function(e) NULL)
  if (is.null(meta)) return(character(0))

  unlist(meta[["r-packages"]], use.names = FALSE)
}

# ------------------------------------------------------------------------------

cli::cli_h1("Building package map")

repo_root <- here::here()
tidymodels_members <- tidymodels::tidymodels_packages()

qmd_files <- list.files(repo_root, pattern = "\\.qmd$", recursive = TRUE, full.names = TRUE)
qmd_files <- qmd_files[!grepl("/_site/|/_freeze/", qmd_files)]

cli::cli_alert_info("Found {length(qmd_files)} .qmd files")

# Build package -> pages mapping
pkg_map <- list()

for (path in qmd_files) {
  rel_path <- sub(paste0(repo_root, "/"), "", path, fixed = TRUE)
  pkgs <- parse_r_packages(path)
  if (length(pkgs) == 0) next

  # Expand tidymodels to its members
  if ("tidymodels" %in% pkgs) {
    pkgs <- union(pkgs, tidymodels_members)
  }

  for (pkg in pkgs) {
    pkg_map[[pkg]] <- sort(unique(c(pkg_map[[pkg]], rel_path)))
  }
}

pkg_map <- pkg_map[sort(names(pkg_map))]

out_path <- here::here("data/package_map.json")
jsonlite::write_json(pkg_map, out_path, pretty = TRUE, auto_unbox = FALSE)

cli::cli_alert_success("Written to {.file package_map.json}")
cli::cli_alert_info("{length(pkg_map)} packages mapped across {length(qmd_files)} files")
