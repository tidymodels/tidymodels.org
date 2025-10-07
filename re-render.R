library(fs)
library(quarto)

# Deletes freeze folder

if (dir_exists("_freeze")) {
  dir_delete("_freeze")
}
if (dir_exists("site_libs")) {
  dir_delete("site_libs")
}

# Deletes all `cache` folders
dir_ls(recurse = TRUE, type = "directory", regexp = "cache") |>
  purrr::walk(dir_delete)

# Update function lists
source("make_function_lists.R")

# Starts rerender
quarto_render()
