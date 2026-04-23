# ------------------------------------------------------------------------------
# Re-render specific pages by path.
# Run: Rscript re-render-pages.R <page1> [page2 ...]
#
# Examples:
#   Rscript re-render-pages.R learn/models/parsnip-nnet/index.qmd
#   Rscript re-render-pages.R learn/models/parsnip-nnet/index.qmd start/resampling/index.qmd

library(cli)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) == 0) {
  cli::cli_abort("Usage: Rscript re-render-pages.R <page1> [page2 ...]")
}

repo_root <- here::here()
pages <- sort(unique(args))

cli::cli_h1("Selective re-render")
cli::cli_alert_info("{length(pages)} page{?s} to re-render")

# ------------------------------------------------------------------------------

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
  t0      <- proc.time()[["elapsed"]]
  result  <- system2("quarto", c("render", file.path(repo_root, page)))
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
