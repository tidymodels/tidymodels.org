# Re-render specific pages by path.
# Run: Rscript re-render-pages.R <page1> [page2 ...]
#
# Examples:
#   Rscript re-render-pages.R learn/models/parsnip-nnet/index.qmd
#   Rscript re-render-pages.R learn/models/parsnip-nnet/index.qmd start/resampling/index.qmd

source(here::here("R/render_helpers.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  cli::cli_abort("Usage: Rscript re-render-pages.R <page1> [page2 ...]")
}

cli::cli_h1("Selective re-render")
render_pages(sort(unique(args)))
