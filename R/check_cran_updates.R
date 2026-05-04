# Check CRAN for package updates and write results to GITHUB_OUTPUT.
# Run: Rscript check_cran_updates.R [manual_override]
#   manual_override: space-separated package names, or "--all", or empty string

library(jsonlite)

write_output <- function(...) {
  cat(..., file = Sys.getenv("GITHUB_OUTPUT"), append = TRUE)
}

recorded <- jsonlite::read_json("data/_versions.json", simplifyVector = TRUE)
pkg_map  <- jsonlite::read_json("data/package_map.json")

resolve_pages <- function(pkgs) {
  if (identical(pkgs, "--all")) {
    pages <- sort(unique(unlist(pkg_map, use.names = FALSE)))
  } else {
    known <- intersect(pkgs, names(pkg_map))
    pages <- sort(unique(unlist(pkg_map[known], use.names = FALSE)))
  }
  paste(pages, collapse = " ")
}

write_output(sprintf("date=%s\n", format(Sys.Date(), "%Y-%m-%d")))

manual <- trimws(commandArgs(trailingOnly = TRUE)[1])
if (is.na(manual)) manual <- ""

if (nzchar(manual)) {
  if (manual == "--all") {
    cat("Manual override: re-rendering all pages\n")
    write_output("updated_packages=--all\n")
    write_output(sprintf("pages=%s\n", resolve_pages("--all")))
    write_output("has_updates=true\n")
    write_output("version_changes<<EOF\n- (all pages, manual trigger)\nEOF\n")
  } else {
    pkgs <- trimws(strsplit(manual, " ")[[1]])
    pkgs <- pkgs[nzchar(pkgs)]
    cat("Manual override for packages:", paste(pkgs, collapse = ", "), "\n")
    changes <- paste0("- ", pkgs, ": ", recorded[pkgs], " (manual trigger)")
    write_output(sprintf("updated_packages=%s\n", paste(pkgs, collapse = " ")))
    write_output(sprintf("pages=%s\n", resolve_pages(pkgs)))
    write_output("has_updates=true\n")
    write_output(sprintf("version_changes<<EOF\n%s\nEOF\n", paste(changes, collapse = "\n")))
  }
  quit(save = "no")
}

pkgs <- names(recorded)

db <- available.packages(repos = "https://cloud.r-project.org")
cran_versions <- setNames(
  ifelse(pkgs %in% rownames(db), db[pkgs[pkgs %in% rownames(db)], "Version"], NA),
  pkgs
)

updated <- pkgs[
  !is.na(cran_versions) &
  package_version(cran_versions[pkgs]) != package_version(unlist(recorded[pkgs]))
]

if (length(updated) == 0) {
  cat("No package updates found.\n")
  write_output("has_updates=false\n")
} else {
  cat("Updated packages:", paste(updated, collapse = ", "), "\n")
  changes <- paste0("- ", updated, ": ", unlist(recorded[updated]), " → ", cran_versions[updated])
  write_output(sprintf("updated_packages=%s\n", paste(updated, collapse = " ")))
  write_output(sprintf("pages=%s\n", resolve_pages(updated)))
  write_output("has_updates=true\n")
  write_output(sprintf("version_changes<<EOF\n%s\nEOF\n", paste(changes, collapse = "\n")))
}
