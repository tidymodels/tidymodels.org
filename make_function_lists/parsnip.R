# ------------------------------------------------------------------------------
# Generate parsnip models reference list
# Run: Rscript make_function_lists/parsnip.R

source(here::here("make_function_lists/_utils.R"))

# ------------------------------------------------------------------------------

parsnip_pkgs <- revdepcheck::cran_revdeps(
  "parsnip",
  dependencies = c("Depends", "Imports")
)
parsnip_pkgs <- c(parsnip_pkgs, "parsnip")

# These ignore the tidymodels design principles and/or don't work with the
# broader ecosystem or we don't don't have any models in them
excl <- c(
  "additive",
  "bayesian",
  "cuda.ml",
  "SSLR",
  "workflowsets",
  "workflows",

  "tune",
  "tidymodels",
  "shinymodels",
  "stacks",
  "viruslearner"
)
parsnip_pkgs <- parsnip_pkgs[!(parsnip_pkgs %in% excl)]

pak::pak(parsnip_pkgs)

# Load them then get the model data base
loaded <- purrr::map_lgl(
  parsnip_pkgs,
  ~ suppressPackageStartupMessages(require(.x, character.only = TRUE))
)
cli::cli_alert_info("Loaded {sum(loaded)}/{length(loaded)} packages")

# h2o overwrites soooo many functions; this may take a few minutes
conflicted::conflict_prefer_all("base", loser = "h2o", quiet = TRUE)

origin_pkg <- rlang::env_get_list(
  env = parsnip::get_model_env(),
  nms = ls(parsnip::get_model_env(), pattern = "_pkg")
) %>%
  purrr::list_rbind(names_to = "model") %>%
  dplyr::mutate(
    pkg = purrr::map_chr(
      pkg,
      ~ {
        pkg <- intersect(.x, parsnip_pkgs)
        if (length(pkg) == 0) {
          pkg <- "parsnip"
        }
        pkg
      }
    )
  ) %>%
  dplyr::mutate(model = stringr::str_remove(model, "_pkgs$"))

model_list <-
  purrr::map_dfr(get_from_env("models"), ~ get_from_env(.x) %>% mutate(model = .x)) %>%
  dplyr::mutate(
    mode = factor(
      mode,
      levels = c("classification", "regression", "censored regression")
    )
  ) %>%
  dplyr::left_join(origin_pkg, by = c("engine", "mode", "model")) %>%
  dplyr::mutate(
    functions = glue("details_{model}_{engine}")
  )

parsnip_model_info <-
  purrr::map_dfr(
    parsnip_pkgs,
    get_pkg_info,
    keep_internal = TRUE,
    .progress = TRUE
  ) %>%
  sort_out_urls()

# Split model/engine combinations by whether they have "details" pages. Link to
# the details pages whenever possible.

has_details <-
  parsnip_model_info %>%
  dplyr::filter(grepl("^details_", functions)) %>%
  dplyr::inner_join(model_list, by = "functions") %>%
  dplyr::mutate(topic = gsub("<tt>details_", "<tt>", topic))

no_details <-
  model_list %>%
  dplyr::anti_join(
    has_details %>% dplyr::select(model, engine),
    by = c("model", "engine")
  ) %>%
  dplyr::mutate(functions = model) %>%
  dplyr::inner_join(parsnip_model_info, by = "functions")

parsnip_models <-
  no_details %>%
  dplyr::select(title, model, engine, topic, mode, package = pkg) %>%
  dplyr::bind_rows(
    has_details %>%
      dplyr::select(title, model, engine, topic, mode, package = pkg)
  ) %>%
  dplyr::mutate(
    model = paste0("<code>", model, "</code>"),
    engine = paste0("<code>", engine, "</code>"),
    title = gsub("General Interface for ", "", title)
  ) %>%
  dplyr::arrange(model, engine) %>%
  dplyr::select(title, model, engine, topic, mode, package)

write_csv(
  parsnip_models,
  file = here::here("find/parsnip/parsnip_models.csv")
)

cli::cli_alert_success("Generated find/parsnip/parsnip_models.csv")
