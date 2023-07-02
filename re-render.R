library(fs)
library(quarto)

# Deletes freeze folder
dir_delete("_freeze")

# Deletes all `cache` folders
dir_ls(recurse = TRUE, type = "directory", regexp = "cache") |>
  dir_delete()

# Starts rerender
quarto_render()
