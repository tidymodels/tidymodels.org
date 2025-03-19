# Don't run manually, this will run automatically after render. 
# As per _quarto.yml

files <- Sys.getenv("QUARTO_PROJECT_OUTPUT_FILES")
files <- stringr::str_split_1(files, "\n")
files <- stringr::str_remove(files, "_site/")
files <- paste0(files, ".md")

remove_double_newlines <- function(file) {
  txt <- readr::read_lines(file)
  txt <- paste(txt, collapse = "\n")
  txt <- stringr::str_replace_all(txt, "\n{2,}", "\n\n")
  readr::write_lines(txt, file)
}

purrr::walk(files, remove_double_newlines)