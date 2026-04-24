library(jsonlite)

s <- jsonlite::read_json("_render_summary.json")

format_dur <- function(sec) {
  if (sec >= 60) sprintf("%dm %ds", floor(sec / 60), round(sec %% 60))
  else sprintf("%ds", round(sec))
}

lines <- vapply(names(s$pages), function(p) {
  sprintf("| `%s` | %s |", p, format_dur(s$pages[[p]]$duration))
}, character(1))

table <- paste0(
  "| Page | Time |\n",
  "|---|---|\n",
  paste(lines, collapse = "\n")
)

footer <- sprintf(
  "\n**Total:** %d page%s in %s",
  s$n_total,
  if (s$n_total == 1) "" else "s",
  format_dur(s$total_sec)
)

body <- paste0(table, footer)
cat("render_table<<EOF\n", body, "\nEOF\n",
    sep = "", file = Sys.getenv("GITHUB_OUTPUT"), append = TRUE)
