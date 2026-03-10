# ------------------------------------------------------------------------------
# Generate tidyclust models reference list
# Run: Rscript make_function_lists/tidyclust.R

source(here::here("make_function_lists/_utils.R"))

# ------------------------------------------------------------------------------

# currently does not rely on any packages, leaving in if added in future
tidyclust_pkgs <- revdepcheck::cran_revdeps(
  "tidyclust",
  dependencies = c("Depends", "Imports")
)

tidyclust_pkgs <- c(tidyclust_pkgs, "tidyclust")

# Load them then get the model data base
loaded <- purrr::map_lgl(
  tidyclust_pkgs,
  ~ suppressPackageStartupMessages(require(.x, character.only = TRUE))
)
if (any(!loaded)) {
  cli::cli_abort("Some tidyclust packages didn't load: {.pkg {tidyclust_pkgs[!loaded]}}.")
}

# h2o overwrites soooo many functions; this may take a few minutes
conflicted::conflict_prefer_all("base", loser = "h2o", quiet = TRUE)

origin_pkg <- rlang::env_get_list(
  env = modelenv::get_model_env(),
  nms = ls(modelenv::get_model_env(), pattern = "_pkg")
) %>%
  purrr::list_rbind(names_to = "model") %>%
  dplyr::mutate(
    pkg = purrr::map_chr(
      pkg,
      ~ {
        pkg <- intersect(.x, tidyclust_pkgs)
        if (length(pkg) == 0) {
          pkg <- "tidyclust"
        }
        pkg
      }
    )
  ) %>%
  dplyr::mutate(
    model = stringr::str_remove(model, "_pkgs$"),
    functions = paste("details", model, engine, sep = "_")
  )

# Add more clustering methods as they become available
cluster_types <- c(
  "Hierarchical \\(Agglomerative\\) Clustering",
  "K-[M|m]eans"
)

tidyclust_model_info <-
  purrr::map_dfr(
    tidyclust_pkgs,
    get_pkg_info,
    keep_internal = TRUE,
    .progress = TRUE
  ) %>%
  sort_out_urls() %>%
  filter(str_detect(title, paste(cluster_types, collapse = "|")))

tidyclust_models <-
  tidyclust_model_info %>%
  dplyr::filter(grepl("^details_", functions)) %>%
  dplyr::inner_join(origin_pkg, by = "functions") %>%
  dplyr::mutate(
    model = paste0("<code>", model, "</code>"),
    engine = paste0("<code>", engine, "</code>"),
    title = gsub("General Interface for ", "", title)
  ) %>%
  dplyr::arrange(model, engine) %>%
  dplyr::select(title, model, engine, topic, mode, package)

write_csv(
  tidyclust_models,
  file = here::here("find/tidyclust/tidyclust_models.csv")
)

cli::cli_alert_success("Generated find/tidyclust/tidyclust_models.csv")
