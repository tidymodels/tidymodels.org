# ------------------------------------------------------------------------------
# Make data sets for function reference searches. Run this offline to refresh
# data objects.

library(tidymodels)
library(glue)
library(utils)
library(revdepcheck)
library(fs)
library(pkgdown)
library(urlchecker)
library(stringr)
library(readr)

# ------------------------------------------------------------------------------

tidymodels_prefer()
theme_set(theme_bw())
options(pillar.advice = FALSE, pillar.min_title_chars = Inf)

# ------------------------------------------------------------------------------
# Use the pkgdown package to parse the source files and put them into a usable
# format

# TODO find a better way to figure out how to find the true "check_" recipe
# operations from just the source files

get_pkg_info <- function(
  pkg,
  pth = tempdir(),
  keep_internal = FALSE,
  pattern = NULL
) {
  src_file <-
    download.packages(
      pkg,
      destdir = pth,
      repos = "https://cran.rstudio.com/",
      quiet = TRUE
    )
  if (nrow(src_file) != length(pkg)) {
    return(NULL)
    rlang::warn(glue::glue("package {pkg} was not downloaded"))
  }
  pkg_path <- fs::path(pth, pkg)
  on.exit(fs::dir_delete(pkg_path))

  untar_res <- purrr::map_int(src_file[, 2], untar, exdir = pth)
  fs::file_delete(src_file[, 2])
  if (any(untar_res != 0)) {
    rlang::abort(glue::glue("package {pkg} did not unpack correctly"))
  }
  pkg_info <- pkgdown::as_pkgdown(pkg_path)
  res <- pkg_info$topics
  if (!keep_internal) {
    res <- dplyr::filter(res, !internal)
  }
  res <-
    res %>%
    dplyr::select(file_out, functions = alias, title) %>%
    tidyr::unnest(functions) %>%
    dplyr::mutate(package = pkg, all_urls = list(pkg_info$desc$get_urls())) %>%
    dplyr::relocate(package, all_urls)
  if (!is.null(pattern)) {
    res <- dplyr::filter(res, grepl(pattern, functions))
  }
  res
}

# See if any of the urls appear to correspond to the _standard_ pkgdown
# structure. Is so, link to the specific pkgdown html package, otherwise link to
# the first url or, if there are none listed, the canonical CRAN page link. We
# use an internal function in urlchecker to essentially ping the potential url

sort_out_urls <- function(x) {
  test_urls <-
    x %>%
    dplyr::group_by(package) %>%
    dplyr::slice(1) %>%
    dplyr::ungroup() %>%
    tidyr::unnest(all_urls) %>%
    dplyr::mutate(
      URL = purrr::map_chr(all_urls, ~ glue("{.x[[1]]}/reference/index.html")),
      URL = gsub("//", "/", URL, fixed = TRUE)
    ) %>%
    dplyr::select(URL, Parent = functions, package, all_urls)
  url_check_fails <-
    urlchecker:::tools$check_url_db(test_urls) %>%
    dplyr::select(URL)
  pkgdown_urls <-
    test_urls %>%
    dplyr::anti_join(url_check_fails, by = "URL") %>%
    dplyr::select(package, pkgdown_url = all_urls) %>%
    dplyr::group_by(package) %>%
    dplyr::slice(1) %>%
    dplyr::ungroup()
  x %>%
    dplyr::left_join(pkgdown_urls, by = "package") %>%
    dplyr::mutate(
      first_url = purrr::map_chr(all_urls, ~ .x[1]),
      first_url = ifelse(
        is.na(first_url),
        glue("https://cran.r-project.org/package={package}"),
        first_url
      ),
      base_url = ifelse(is.na(pkgdown_url), first_url, pkgdown_url),
      url = ifelse(
        !is.na(pkgdown_url),
        glue("{pkgdown_url}/reference/{file_out}"),
        base_url
      ),
      topic = glue("<a href='{url}' target='_blank'><tt>{functions}</tt></a>")
    ) %>%
    dplyr::select(title, functions, topic, package) %>%
    dplyr::mutate(package = as.factor(package)) %>%
    dplyr::filter(!grepl("deprecated", tolower(title))) %>%
    dplyr::arrange(tolower(gsub("[[:punct:]]", "", title)))
}

# ------------------------------------------------------------------------------

broom_pkgs <- revdepcheck::cran_revdeps(
  "broom",
  dependencies = c("Depends", "Imports")
)
generics_pkgs <- revdepcheck::cran_revdeps("generics", dependencies = "Imports")

broom_pkgs <- sort(unique(c(broom_pkgs, generics_pkgs)))
excl <- c("hydrorecipes", "healthcareai")
broom_pkgs <- broom_pkgs[!(broom_pkgs %in% excl)]

broom_functions <-
  purrr::map_dfr(
    broom_pkgs,
    get_pkg_info,
    pattern = "(^tidy\\.)|(^glance\\.)|(^augment\\.)",
    .progress = TRUE
  ) %>%
  sort_out_urls() %>%
  dplyr::select(-functions)

write_csv(
  broom_functions,
  file = "find/broom/broom_functions.csv"
)

# ------------------------------------------------------------------------------

recipe_pkgs <- revdepcheck::cran_revdeps(
  "recipes",
  dependencies = c("Depends", "Imports")
)
recipe_pkgs <- c(recipe_pkgs, "recipes")

recipe_pkgs <- sort(unique(c(recipe_pkgs)))
excl <- c("hydrorecipes", "healthcareai")
recipe_pkgs <- recipe_pkgs[!(recipe_pkgs %in% excl)]

recipe_functions <-
  purrr::map_dfr(
    recipe_pkgs,
    get_pkg_info,
    pattern = "^step_",
    .progress = TRUE
  ) %>%
  sort_out_urls() %>%
  dplyr::select(-functions)

write_csv(
  recipe_functions,
  file = "find/recipes/recipe_functions.csv"
)

# ------------------------------------------------------------------------------

all_tm <- read_csv("all_packages.csv", show_col_types = FALSE)$name

tidymodels_functions <-
  purrr::map_dfr(
    all_tm,
    get_pkg_info,
    .progress = TRUE
  ) %>%
  sort_out_urls() %>%
  dplyr::filter(!grepl("^\\.", functions)) %>%
  dplyr::select(-functions)

write_csv(
  tidymodels_functions,
  file = "find/all/tidymodels_functions.csv"
)

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
table(loaded)

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
  file = "find/parsnip/parsnip_models.csv"
)

# ------------------------------------------------------------------------------

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
  file = "find/sparse/models.csv"
)

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
  file = "find/sparse/steps_generate.csv"
)

sparse_steps_preserve <- .S3methods(".recipes_preserve_sparsity") %>%
  as.character() %>%
  stringr::str_remove(".recipes_preserve_sparsity.") %>%
  setdiff(c("default", "recipe")) %>%
  dplyr::as_tibble() %>%
  dplyr::left_join(recipe_functions_with_names, by = c("value" = "name"))

write_csv(
  sparse_steps_preserve,
  file = "find/sparse/steps_preserve.csv"
)

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
table(loaded)

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
  parsnip_models,
  file = "find/tidyclust/tidyclust_models.csv"
)
