# ------------------------------------------------------------------------------
# Shared utilities for generating function reference lists
# Source this file before running any individual generator

suppressPackageStartupMessages({
library(tidymodels)
library(glue)
library(utils)
library(revdepcheck)
library(fs)
library(desc)
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
# Persistent cache for parsed package info (CSV for diff-friendly storage)
# Single file stores all packages with their versions

pkg_cache_file <- here::here("make_function_lists/.pkg_cache.csv")

# In-memory cache (loaded once per session)
pkg_cache <- NULL

# Convert list column to/from pipe-separated string for CSV storage
urls_to_string <- function(urls) {
  purrr::map_chr(urls, ~ paste(.x, collapse = "|"))
}

string_to_urls <- function(s) {
  purrr::map(s, ~ {
    if (is.na(.x) || .x == "" || .x == "NA") {
      NA_character_
    } else {
      strsplit(.x, "\\|")[[1]]
    }
  })
}

load_pkg_cache <- function() {
  if (!is.null(pkg_cache)) return(pkg_cache)

  if (fs::file_exists(pkg_cache_file)) {
    pkg_cache <<- readr::read_csv(
      pkg_cache_file,
      col_types = readr::cols(
        package = readr::col_character(),
        version = readr::col_character(),
        file_out = readr::col_character(),
        functions = readr::col_character(),
        title = readr::col_character(),
        internal = readr::col_logical(),
        all_urls = readr::col_character()
      ),
      show_col_types = FALSE
    ) |>
      dplyr::mutate(all_urls = string_to_urls(all_urls))
  } else {
    pkg_cache <<- tibble::tibble(
      package = character(),
      version = character(),
      file_out = character(),
      functions = character(),
      title = character(),
      internal = logical(),
      all_urls = list()
    )
  }
  pkg_cache
}

save_pkg_cache <- function() {
  if (!is.null(pkg_cache)) {
    pkg_cache |>
      dplyr::mutate(all_urls = urls_to_string(all_urls)) |>
      readr::write_csv(pkg_cache_file)
  }
}

get_pkg_version <- function(pkg, available = NULL) {
  if (is.null(available)) {
    available <- get_available_packages()
  }
  idx <- match(pkg, available[, "Package"])
  if (is.na(idx)) return(NA_character_)
  available[idx, "Version"]
}

get_cached_pkg_info <- function(pkg, version) {
  cache <- load_pkg_cache()
  cached <- dplyr::filter(cache, package == !!pkg, version == !!version)
  if (nrow(cached) > 0) return(cached)
  NULL
}

set_cached_pkg_info <- function(pkg, version, result) {
  load_pkg_cache()
  # Remove old entries for this package
  pkg_cache <<- dplyr::filter(pkg_cache, package != !!pkg)
  # Add new entries
  pkg_cache <<- dplyr::bind_rows(pkg_cache, result)
  save_pkg_cache()
}

clear_pkg_cache <- function() {
  pkg_cache <<- NULL
  if (fs::file_exists(pkg_cache_file)) {
    fs::file_delete(pkg_cache_file)
  }
}

# ------------------------------------------------------------------------------
# Parse source package Rd files directly (faster than pkgdown)
# Uses caching to avoid re-downloading unchanged packages

get_pkg_info <- function(
    pkg,
    pth = tempdir(),
    keep_internal = FALSE,
    pattern = NULL,
    available = NULL
) {
  # Get current version from CRAN
  version <- get_pkg_version(pkg, available)
  if (is.na(version)) {
    rlang::warn(glue::glue("package {pkg} not found in available packages"))
    return(NULL)
  }

  # Check cache first
  cached <- get_cached_pkg_info(pkg, version)
  if (!is.null(cached)) {
    # Apply keep_internal and pattern filters to cached result
    res <- cached
    if (!keep_internal) {
      res <- dplyr::filter(res, !internal)
    }
    res <- dplyr::select(res, -internal, -version)
    if (!is.null(pattern)) {
      res <- dplyr::filter(res, grepl(pattern, functions))
    }
    return(res)
  }

  # Not cached - download and parse
  src_file <- download.packages(
    pkg,
    destdir = pth,
    repos = "https://cran.rstudio.com/",
    quiet = TRUE,
    available = available
  )
  if (nrow(src_file) != length(pkg)) {
    rlang::warn(glue::glue("package {pkg} was not downloaded"))
    return(NULL)
  }

  pkg_path <- fs::path(pth, pkg)
  on.exit(fs::dir_delete(pkg_path))

  untar(src_file[, 2], exdir = pth)
  fs::file_delete(src_file[, 2])

  # Get URLs from DESCRIPTION (always return as list for consistent column type)
  desc_path <- fs::path(pkg_path, "DESCRIPTION")
  pkg_urls <- if (fs::file_exists(desc_path)) {
    d <- desc::desc(desc_path)
    urls <- d$get_urls()
    if (length(urls) == 0) list(NA_character_) else list(urls)
  } else {
    list(NA_character_)
  }

  # Parse Rd files directly
  man_path <- fs::path(pkg_path, "man")
  if (!fs::dir_exists(man_path)) {
    return(tibble::tibble(
      package = pkg,
      all_urls = pkg_urls,
      file_out = character(),
      functions = character(),
      title = character()
    ))
  }

  rd_files <- fs::dir_ls(man_path, glob = "*.Rd")

  parse_rd_file <- function(rd_file) {
    tryCatch({
      rd <- tools::parse_Rd(rd_file)

      get_tag <- function(rd, tag_name) {
        for (el in rd) {
          tag <- attr(el, "Rd_tag")
          if (!is.null(tag) && tag == tag_name) {
            return(paste(unlist(el), collapse = ""))
          }
        }
        NA_character_
      }

      get_aliases <- function(rd) {
        aliases <- c()
        for (el in rd) {
          tag <- attr(el, "Rd_tag")
          if (!is.null(tag) && tag == "\\alias") {
            aliases <- c(aliases, trimws(paste(unlist(el), collapse = "")))
          }
        }
        aliases
      }

      is_internal <- function(rd) {
        for (el in rd) {
          tag <- attr(el, "Rd_tag")
          if (!is.null(tag) && tag == "\\keyword") {
            kw <- trimws(paste(unlist(el), collapse = ""))
            if (kw == "internal") return(TRUE)
          }
        }
        FALSE
      }

      name <- get_tag(rd, "\\name")
      title <- get_tag(rd, "\\title")
      aliases <- get_aliases(rd)
      internal <- is_internal(rd)

      if (length(aliases) == 0) aliases <- name

      # Clean title: collapse newlines and multiple spaces
      clean_title <- title |>
        gsub("[\r\n]+", " ", x = _) |>
        gsub("\\s+", " ", x = _) |>
        trimws()

      # Use Rd filename for file_out (pkgdown uses filename, not \name tag)
      rd_basename <- fs::path_ext_remove(fs::path_file(rd_file))

      tibble::tibble(
        file_out = paste0(rd_basename, ".html"),
        functions = aliases,
        title = clean_title,
        internal = internal
      )
    }, error = function(e) NULL)
  }

  res <- purrr::map(rd_files, parse_rd_file) |>
    purrr::compact() |>
    purrr::list_rbind()

  if (nrow(res) == 0) {
    empty_result <- tibble::tibble(
      package = pkg,
      version = version,
      file_out = character(),
      functions = character(),
      title = character(),
      internal = logical(),
      all_urls = pkg_urls
    )
    set_cached_pkg_info(pkg, version, empty_result)
    return(dplyr::select(empty_result, -internal, -version))
  }

  # Prepare full result with version and internal columns for caching
  res <- res |>
    tidyr::unnest(functions) |>
    dplyr::mutate(package = pkg, version = version, all_urls = pkg_urls) |>
    dplyr::relocate(package, version, all_urls)

  # Cache full result before filtering
  set_cached_pkg_info(pkg, version, res)

  # Apply filters for this call
  if (!keep_internal) {
    res <- dplyr::filter(res, !internal)
  }
  res <- dplyr::select(res, -internal, -version)

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
