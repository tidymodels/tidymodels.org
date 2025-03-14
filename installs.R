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
  "kableExtra",
  "keras",
  "kernlab",
  "klaR",
  "leaflet",
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
  "rlang",
  "ROSE",
  "rpart.plot",
  "rsample",
  "rstanarm",
  "rules",
  "scales",
  "sessioninfo",
  "readmission",
  "skimr",
  "spatialsample",
  "stacks",
  "stopwords",
  "stringr",
  "textrecipes",
  "text2vec",
  "themis",
  "tidymodels",
  "tidyposterior",
  "timetk",
  "torch",
  "tune",
  "vip",
  "zoo",
  "sweep",
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

