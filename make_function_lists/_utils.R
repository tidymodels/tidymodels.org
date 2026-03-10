# ------------------------------------------------------------------------------
# Shared utilities for generating function reference lists
# Source this file before running any individual generator

suppressPackageStartupMessages({
library(tidymodels)
library(glue)
library(utils)
library(revdepcheck)
library(fs)
library(pkgdown)
library(urlchecker)
library(stringr)
library(readr)
})

# ------------------------------------------------------------------------------

tidymodels_prefer()
theme_set(theme_bw())
options(pillar.advice = FALSE, pillar.min_title_chars = Inf)

# ------------------------------------------------------------------------------
# Cache available packages to avoid repeated CRAN metadata fetches

get_available_packages <- function() {
  available.packages(repos = "https://cran.rstudio.com/")
}

# ------------------------------------------------------------------------------
# Use the pkgdown package to parse the source files and put them into a usable
# format

get_pkg_info <- function(
    pkg,
    pth = tempdir(),
    keep_internal = FALSE,
    pattern = NULL,
    available = NULL
) {
  src_file <-
    download.packages(
      pkg,
      destdir = pth,
      repos = "https://cran.rstudio.com/",
      quiet = TRUE,
      available = available
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
