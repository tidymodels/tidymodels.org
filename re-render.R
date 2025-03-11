library(fs)
library(quarto)

# Deletes freeze folder
dir_delete("_freeze")
dir_delete("site_libs")

# Deletes all `cache` folders
dir_ls(recurse = TRUE, type = "directory", regexp = "cache") |>
  dir_delete()

# Update function lists
source("make_function_lists.R")

# Starts rerender
quarto_render()
