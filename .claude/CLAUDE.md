@../README.md

## Instructions

Always update README.md when any information in it becomes out of date (file paths, script names, workflow steps, etc.).

After editing any `r-packages:` field in a `.qmd` front matter (adding, removing, or renaming a package), run `Rscript R/make_package_map.R` to regenerate `data/package_map.json`. Do this without asking — it's a local, reversible action.
