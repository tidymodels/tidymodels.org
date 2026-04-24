# ------------------------------------------------------------------------------
# Install R packages needed by the given .qmd pages (reads r-packages: front matter).
# Run: Rscript install_for_pages.R <page1.qmd> [page2.qmd ...]

library(cli)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) == 0) {
  cli::cli_abort("Usage: Rscript install_for_pages.R <page1.qmd> [page2.qmd ...]")
}

repo_root <- here::here()

# ------------------------------------------------------------------------------
# Read r-packages: front matter from each page

needed <- character(0)
for (page in args) {
  lines <- readLines(page, warn = FALSE)
  in_block <- FALSE
  for (line in lines) {
    if (grepl("^r-packages:", line)) { in_block <- TRUE; next }
    if (in_block && grepl("^  - ", line)) {
      needed <- c(needed, trimws(sub("^  - ", "", line)))
    } else if (in_block) {
      break
    }
  }
}
needed <- sort(unique(needed))

source(file.path(repo_root, "R/install_packages.R"))
install_packages(needed)
