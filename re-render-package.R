# ------------------------------------------------------------------------------
# Selectively re-render pages that depend on one or more R packages.
# Run: Rscript re-render-package.R <pkg1> [pkg2 ...]
#
# Examples:
#   Rscript re-render-package.R ranger
#   Rscript re-render-package.R ranger glmnet
#   Rscript re-render-package.R tidymodels   # re-renders all tidymodels pages
#   Rscript re-render-package.R --all        # re-renders every page

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
cli::cli_alert_info("{length(pages)} page{?s} to re-render")

# ------------------------------------------------------------------------------
# Clear freeze cache for affected pages, then render

failed    <- character(0)
durations <- numeric(0)

for (page in pages) {
  rel_dir    <- dirname(page)
  freeze_dir <- file.path(repo_root, "_freeze", rel_dir)

  if (dir.exists(freeze_dir)) {
    unlink(freeze_dir, recursive = TRUE)
    cli::cli_alert("Cleared freeze cache: {.path {freeze_dir}}")
  }

  cli::cli_h2("Rendering {.file {page}}")
  t0     <- proc.time()[["elapsed"]]
  result <- system2("quarto", c("render", file.path(repo_root, page)))
  elapsed <- proc.time()[["elapsed"]] - t0
  durations <- c(durations, setNames(elapsed, page))

  if (result != 0) {
    failed <- c(failed, page)
    cli::cli_alert_danger("Render failed for {.file {page}}")
  }
}

# Write render summary for use by CI (e.g. PR body)
summary_path <- file.path(repo_root, "_render_summary.json")
jsonlite::write_json(
  list(
    pages     = as.list(setNames(
      lapply(names(durations), function(p) list(
        duration = round(durations[[p]], 1)
      )),
      names(durations)
    )),
    n_total   = length(pages),
    n_failed  = length(failed),
    total_sec = round(sum(durations), 1)
  ),
  summary_path,
  auto_unbox = TRUE,
  pretty     = TRUE
)

if (length(failed) > 0) {
  cli::cli_h2("Summary")
  cli::cli_alert_success("{length(pages) - length(failed)} page{?s} rendered successfully.")
  cli::cli_alert_danger("{length(failed)} page{?s} failed:")
  for (page in failed) cli::cli_bullets(c("x" = page))
  quit(save = "no", status = 1)
}

cli::cli_alert_success("Done. {length(pages)} page{?s} re-rendered successfully.")
