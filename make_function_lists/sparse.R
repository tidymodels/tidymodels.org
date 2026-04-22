# ------------------------------------------------------------------------------
# Generate sparse support reference lists
# Run: Rscript make_function_lists/sparse.R
#
# Note: This depends on parsnip_models.csv and recipe_functions.csv
# Run parsnip.R and recipes.R first if those files don't exist.

source(here::here("make_function_lists/_utils.R"))

# ------------------------------------------------------------------------------
# Check dependencies exist

parsnip_csv <- here::here("find/parsnip/parsnip_models.csv")
recipes_csv <- here::here("find/recipes/recipe_functions.csv")

if (!file.exists(parsnip_csv)) {

  cli::cli_abort(c(
    "Missing {.file find/parsnip/parsnip_models.csv}",
    "i" = "Run {.code Rscript make_function_lists/parsnip.R} first"
  ))
}

if (!file.exists(recipes_csv)) {
  cli::cli_abort(c(
    "Missing {.file find/recipes/recipe_functions.csv}",
    "i" = "Run {.code Rscript make_function_lists/recipes.R} first"
  ))
}

# ------------------------------------------------------------------------------
# Load parsnip packages to access model environment

parsnip_pkgs <- revdepcheck::cran_revdeps(
  "parsnip",
  dependencies = c("Depends", "Imports")
)
parsnip_pkgs <- c(parsnip_pkgs, "parsnip")

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
  "viruslearner",
  "nestedmodels",
  "viraldomain",
  "viralmodels"
)
parsnip_pkgs <- parsnip_pkgs[!(parsnip_pkgs %in% excl)]

loaded <- purrr::map_lgl(
  parsnip_pkgs,
  ~ suppressPackageStartupMessages(require(.x, character.only = TRUE))
)

conflicted::conflict_prefer_all("base", loser = "h2o", quiet = TRUE)

# ------------------------------------------------------------------------------
# Sparse models

parsnip_models <- read_csv(parsnip_csv, show_col_types = FALSE)

sparse_models <- rlang::env_get_list(
  env = parsnip::get_model_env(),
  nms = ls(parsnip::get_model_env(), pattern = "_encoding")
) %>%
  purrr::list_rbind(names_to = "model") %>%
  dplyr::filter(allow_sparse_x) %>%
  dplyr::distinct(model, engine) %>%
  dplyr::filter(stringr::str_detect(engine, "_offset$", negate = TRUE)) %>%
  dplyr::mutate(model = stringr::str_remove(model, "_encoding$")) %>%
  dplyr::left_join(
    by = dplyr::join_by(model, engine),
    parsnip_models %>%
      dplyr::mutate(
        dplyr::across(
          c(model, engine),
          \(x) stringr::str_remove_all(x, "(<code>|</code>)")
        )
      )
  ) %>%
  dplyr::select(model, engine, topic) %>%
  dplyr::distinct()

write_csv(
  sparse_models,
  file = here::here("find/sparse/models.csv")
)

cli::cli_alert_success("Generated find/sparse/models.csv")

# ------------------------------------------------------------------------------
# Sparse recipe steps

recipe_functions <- read_csv(recipes_csv, show_col_types = FALSE)

recipe_functions_with_names <- recipe_functions %>%
  dplyr::mutate(
    name = stringr::str_extract(topic, "<tt>.*"),
    name = stringr::str_remove(name, "<tt>"),
    name = stringr::str_remove(name, "</tt></a>")
  )

library(textrecipes)
library(embed)
library(extrasteps)

sparse_steps_generate <- .S3methods(".recipes_estimate_sparsity") %>%
  as.character() %>%
  stringr::str_remove(".recipes_estimate_sparsity.") %>%
  setdiff(c("default", "recipe")) %>%
  dplyr::as_tibble() %>%
  dplyr::left_join(recipe_functions_with_names, by = c("value" = "name"))

write_csv(
  sparse_steps_generate,
  file = here::here("find/sparse/steps_generate.csv")
)

cli::cli_alert_success("Generated find/sparse/steps_generate.csv")

sparse_steps_preserve <- .S3methods(".recipes_preserve_sparsity") %>%
  as.character() %>%
  stringr::str_remove(".recipes_preserve_sparsity.") %>%
  setdiff(c("default", "recipe")) %>%
  dplyr::as_tibble() %>%
  dplyr::left_join(recipe_functions_with_names, by = c("value" = "name"))

write_csv(
  sparse_steps_preserve,
  file = here::here("find/sparse/steps_preserve.csv")
)

cli::cli_alert_success("Generated find/sparse/steps_preserve.csv")
