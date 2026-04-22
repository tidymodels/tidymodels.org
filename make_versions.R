# ------------------------------------------------------------------------------
# Record the current installed versions of all packages in package_map.json.
# Run this after a re-render to update _versions.json.
# Run: Rscript make_versions.R

library(jsonlite)
library(cli)

repo_root <- here::here()
map_path  <- file.path(repo_root, "package_map.json")

if (!file.exists(map_path)) {
  cli::cli_abort(
    "{.file package_map.json} not found. Run {.code Rscript make_package_map.R} first."
  )
}

pkgs <- names(jsonlite::read_json(map_path))

versions <- lapply(pkgs, function(pkg) {
  tryCatch(
    as.character(packageVersion(pkg)),
    error = function(e) NA_character_
  )
})
names(versions) <- pkgs

# Drop packages that aren't installed
versions <- Filter(Negate(is.na), versions)
versions <- versions[sort(names(versions))]

out_path <- file.path(repo_root, "_versions.json")
jsonlite::write_json(versions, out_path, pretty = TRUE, auto_unbox = TRUE)

cli::cli_alert_success(
  "Recorded versions for {length(versions)} package{?s} in {.file _versions.json}"
)
