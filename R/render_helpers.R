# Shared helpers for re-render scripts.

library(cli)

# Clear the freeze cache for a page and render it with quarto.
# Returns a named numeric of elapsed seconds, with the page path as name.
render_page <- function(page, repo_root = here::here()) {
  freeze_dir <- file.path(repo_root, "_freeze", dirname(page))
  if (dir.exists(freeze_dir)) {
    unlink(freeze_dir, recursive = TRUE)
    cli::cli_alert("Cleared freeze cache: {.path {freeze_dir}}")
  }

  cli::cli_h2("Rendering {.file {page}}")
  t0      <- proc.time()[["elapsed"]]
  result  <- system2("quarto", c("render", file.path(repo_root, page)))
  elapsed <- proc.time()[["elapsed"]] - t0

  list(page = page, elapsed = elapsed, failed = result != 0)
}

# Render a character vector of pages and write _render_summary.json.
# Exits with status 1 if any page fails.
render_pages <- function(pages, repo_root = here::here()) {
  cli::cli_alert_info("{length(pages)} page{?s} to re-render")

  results   <- lapply(pages, render_page, repo_root = repo_root)
  failed    <- Filter(function(r) r$failed, results)
  durations <- setNames(
    sapply(results, `[[`, "elapsed"),
    sapply(results, `[[`, "page")
  )

  jsonlite::write_json(
    list(
      pages     = as.list(setNames(
        lapply(names(durations), function(p) list(duration = round(durations[[p]], 1))),
        names(durations)
      )),
      n_total   = length(pages),
      n_failed  = length(failed),
      total_sec = round(sum(durations), 1)
    ),
    file.path(repo_root, "_render_summary.json"),
    auto_unbox = TRUE,
    pretty     = TRUE
  )

  if (length(failed) > 0) {
    cli::cli_h2("Summary")
    cli::cli_alert_success("{length(pages) - length(failed)} page{?s} rendered successfully.")
    cli::cli_alert_danger("{length(failed)} page{?s} failed:")
    for (r in failed) cli::cli_bullets(c("x" = r$page))
    quit(save = "no", status = 1)
  }

  cli::cli_alert_success("Done. {length(pages)} page{?s} re-rendered successfully.")
}
