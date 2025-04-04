packages <- c(
  "abind",
  "AmesHousing",
  "aorsf",
  "AppliedPredictiveModeling",
  "baguette",
  "betacal",
  "blogdown",
  "broom.mixed",
  "brulee",
  "censored",
  "desirability2",
  "detectors",
  "devtools",
  "dials",
  "discrim",
  "doMC",
  "doParallel",
  "dotwhisker",
  "embed",
  "forecast",
  "fs",
  "furrr",
  "future",
  "GGally",
  "glmnet",
  "glue",
  "htmlwidgets",
  "kableExtra",
  "keras",
  "kernlab",
  "klaR",
  "leaflet",
  "lobstr",
  "mda",
  "mlbench",
  "modeldata",
  "modeldatatoo",
  "modeltime",
  "nycflights13",
  "parsnip",
  "partykit",
  "pkgdown",
  "pls",
  "plsmod",
  "poissonreg",
  "probably",
  "prodlim",
  "quantregForest",
  "randomForest",
  "ranger",
  "readmission",
  "rlang",
  "ROSE",
  "rpart.plot",
  "rsample",
  "rstanarm",
  "rules",
  "scales",
  "sessioninfo",
  "skimr",
  "sparsevctrs",
  "spatialsample",
  "stacks",
  "stopwords",
  "stringr",
  "sweep",
  "text2vec",
  "textrecipes",
  "themis",
  "tidymodels",
  "tidyposterior",
  "timetk",
  "torch",
  "tune",
  "vip",
  "zoo",
  "rstudio/DT"
)

pak::pak(packages)

# Manually check for dev versions
# sub("rstudio/", "", packages) |>
#   setdiff(c()) |>
#   sapply(packageVersion)

# Running `library(brulee)` will trigger installation of torch
library(brulee)

# Setup keras
library(keras)

install_keras(method = "virtualenv")

