print(here::here())

knitr::opts_chunk$set(
  digits = 3,
  comment = "#>",
  dev = 'svglite',
  dev.args = list(bg = "transparent"),
  fig.path = "figs/",
  fig.align = "center",
  collapse = TRUE
)
options(width = 80, cli.width = 70)

# A theme-agnostic grey ("#7C7C7C") gives ~4.2:1 contrast against both a white
# page and the site's dark surface (#1A162D), so plot chrome stays legible
# without needing separate light/dark renders of every figure.
theme_tidymodels_transparent <- function(...) {
  ggplot2::theme_bw(...) +
    ggplot2::theme(
      panel.background = ggplot2::element_blank(),
      plot.background = ggplot2::element_blank(),
      legend.background = ggplot2::element_blank(),
      legend.key = ggplot2::element_blank(),
      strip.background = ggplot2::element_blank(),
      panel.grid = ggplot2::element_line(colour = "#7C7C7C", linewidth = 0.2),
      axis.text = ggplot2::element_text(colour = "#7C7C7C"),
      axis.title = ggplot2::element_text(colour = "#7C7C7C"),
      axis.ticks = ggplot2::element_line(colour = "#7C7C7C"),
      plot.title = ggplot2::element_text(colour = "#7C7C7C"),
      plot.subtitle = ggplot2::element_text(colour = "#7C7C7C"),
      strip.text = ggplot2::element_text(colour = "#7C7C7C"),
      legend.text = ggplot2::element_text(colour = "#7C7C7C"),
      legend.title = ggplot2::element_text(colour = "#7C7C7C")
    )
}

# Unmapped geoms (e.g. geom_line() with no `colour` aes) default to black,
# which is as unreadable on the dark surface as the theme colours above.
# Recolor those defaults to the same theme-agnostic grey.
local({
  colour_geoms <- c(
    "line", "path", "point", "step", "segment", "text", "label",
    "abline", "hline", "vline", "boxplot", "errorbar", "linerange",
    "pointrange", "curve", "function", "rug", "density", "freqpoly"
  )
  fill_geoms <- c("bar", "col", "histogram", "area", "ribbon", "violin")
  for (g in colour_geoms) {
    tryCatch(
      ggplot2::update_geom_defaults(g, list(colour = "#7C7C7C")),
      error = function(e) NULL
    )
  }
  for (g in fill_geoms) {
    tryCatch(
      ggplot2::update_geom_defaults(g, list(fill = "#7C7C7C")),
      error = function(e) NULL
    )
  }
  ggplot2::update_geom_defaults("bar", list(colour = NA))
  ggplot2::update_geom_defaults("col", list(colour = NA))
  ggplot2::update_geom_defaults("histogram", list(colour = NA))
})

# Same treatment for base R graphics (rpart.plot(), partykit, survfit plots).
par_transparent <- function() {
  graphics::par(
    bg = NA,
    fg = "#7C7C7C",
    col = "#7C7C7C",
    col.axis = "#7C7C7C",
    col.lab = "#7C7C7C",
    col.main = "#7C7C7C",
    col.sub = "#7C7C7C"
  )
}

article_req_pkgs <- function(x, what = "To use code in this article, ") {
  x <- sort(x)
  x <- knitr::combine_words(x, and = " and ")
  paste0(
    what,
    " you will need to install the following packages: ",
    x,
    "."
  )
}
small_session <- function(pkgs = NULL) {
  pkgs <- c(
    pkgs,
    "recipes",
    "parsnip",
    "tune",
    "workflows",
    "dials",
    "dplyr",
    "broom",
    "ggplot2",
    "purrr",
    "rlang",
    "rsample",
    "tibble",
    "infer",
    "yardstick",
    "tidymodels",
    "infer"
  )
  pkgs <- unique(pkgs)
  library(sessioninfo)
  library(dplyr)
  sinfo <- sessioninfo::session_info()
  cls <- class(sinfo$packages)
  sinfo$packages <-
    sinfo$packages %>%
    dplyr::filter(package %in% pkgs)
  class(sinfo$packages) <- cls

  remove_double_newlines <- function(x) {
    ind <- x == ""
    count <- 0
    for (i in seq_along(ind)) {
      if (ind[i]) {
        count <- count + 1
        if (count == 1) {
          ind[i] <- FALSE
        }
      } else {
        count <- 0
      }
    }
    x[!ind]
  }

  sinfo <- capture.output(sinfo)

  sinfo <- sinfo |>
    stringr::str_subset("^ \\[\\d+\\] ", negate = TRUE) |>
    stringr::str_subset(
      "^ (setting|os|system|ui|collate|ctype|tz|date)",
      negate = TRUE
    ) |>
    stringr::str_remove(" @ .*") |>
    stringr::str_replace_all("\\*", " ") |>
    stringr::str_remove("\\s+(lib\\s+)?source$") |>
    stringr::str_replace(" \\[\\d+\\] ", " ") |>
    stringr::str_remove("\\s+(CRAN|RSPM|Bioconductor|Github|local)(\\s+\\(.*\\))?\\s*$") |>
    stringr::str_subset(
      "Packages attached to the search path",
      negate = TRUE
    ) |>
    remove_double_newlines()
  
  cat(sinfo, sep = "\n")
}
