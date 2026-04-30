# ------------------------------------------------------------------------------
# Install R packages needed by the given .qmd pages (reads r-packages: front matter).
# Run: Rscript install_for_pages.R <page1.qmd> [page2.qmd ...]
#
# Writes two GitHub Actions outputs (when $GITHUB_OUTPUT is set):
#   pages         - space-separated pages that can be rendered (all packages installed)
#   skipped_pages - space-separated pages skipped due to install failures

library(cli)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) == 0) {
  cli::cli_abort("Usage: Rscript install_for_pages.R <page1.qmd> [page2.qmd ...]")
}

repo_root <- here::here()

# ------------------------------------------------------------------------------
# Read r-packages: front matter from each page

page_pkgs <- list()
for (page in args) {
  pkgs <- character(0)
  lines <- readLines(page, warn = FALSE)
  in_block <- FALSE
  for (line in lines) {
    if (grepl("^r-packages:", line)) { in_block <- TRUE; next }
    if (in_block && grepl("^  - ", line)) {
      pkgs <- c(pkgs, trimws(sub("^  - ", "", line)))
    } else if (in_block) {
      break
    }
  }
  page_pkgs[[page]] <- pkgs
}

needed <- sort(unique(unlist(page_pkgs, use.names = FALSE)))

source(file.path(repo_root, "R/install_packages.R"))
failed_pkgs <- install_packages(needed)

# ------------------------------------------------------------------------------
# Filter pages whose required packages all installed successfully

if (length(failed_pkgs) > 0) {
  needs_failed <- vapply(page_pkgs, function(pkgs) any(pkgs %in% failed_pkgs), logical(1))
  skipped <- names(page_pkgs)[needs_failed]
  renderable <- names(page_pkgs)[!needs_failed]

  cli::cli_warn(c(
    "!" = "Skipping {length(skipped)} page{?s} because required package{?s} failed to install:",
    " " = "{.pkg {failed_pkgs}}",
    " " = "{.file {skipped}}"
  ))
} else {
  renderable <- args
  skipped <- character(0)
}

# ------------------------------------------------------------------------------
# Write outputs for GitHub Actions

gha_output <- Sys.getenv("GITHUB_OUTPUT")
if (nzchar(gha_output)) {
  cat(
    sprintf("pages=%s\n", paste(renderable, collapse = " ")),
    sprintf("skipped_pages=%s\n", paste(skipped, collapse = " ")),
    file = gha_output, append = TRUE
  )
}
