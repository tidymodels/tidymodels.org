# Extract r-packages: front matter from .qmd pages and write to GITHUB_OUTPUT.
# Run: Rscript extract_page_packages.R <page1.qmd> [page2.qmd ...]

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("Usage: Rscript extract_page_packages.R <page1.qmd> [page2.qmd ...]")
}

pkgs <- character()
for (p in args) {
  lines <- tryCatch(readLines(p, warn = FALSE), error = function(e) {
    message("Warning: could not read ", p, ": ", e$message)
    character()
  })
  if (length(lines) == 0 || lines[1] != "---") next
  end <- which(lines[-1] == "---")[1] + 1L
  if (is.na(end)) next
  front <- lines[seq(2, end - 1)]
  in_rpkgs <- FALSE
  for (line in front) {
    if (grepl("^r-packages:", line)) {
      in_rpkgs <- TRUE
    } else if (in_rpkgs && grepl("^\\s+-\\s+\\S", line)) {
      pkgs <- c(pkgs, trimws(sub("^\\s*-\\s*", "", line)))
    } else if (in_rpkgs && nzchar(line) && !grepl("^\\s", line)) {
      in_rpkgs <- FALSE
    }
  }
}

result <- paste(sort(unique(pkgs)), collapse = " ")
cat("Extracted packages:", result, "\n")
cat(sprintf("packages=%s\n", result), file = Sys.getenv("GITHUB_OUTPUT"), append = TRUE)
