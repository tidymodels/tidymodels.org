---
title: "Fitting and predicting with parsnip"
categories:
  - model fitting
  - parsnip
  - regression
  - classification
type: learn-subsection
weight: 1
description: | 
  Examples that show how to fit and predict with different combinations of model, mode, and engine.
toc: true
toc-depth: 3
include-after-body: ../../../resources.html
format:
  html:
    theme: ["style.scss"]
---

# Introduction

This page shows examples of how to *fit* and *predict* with different combinations of model, mode, and engine. As a reminder, in parsnip, 

- the **model type** differentiates basic modeling approaches, such as random forests, logistic regression, linear support vector machines, etc.,

- the **mode** denotes in what kind of modeling context it will be used (most commonly, classification or regression), and

- the computational **engine** indicates how the model is fit, such as with a specific R package implementation or even methods outside of R like Keras or Stan.

We'll break the examples up by their mode. For each model, we'll show different data sets used across the different engines. 

To use code in this article,  you will need to install the following packages: agua, baguette, bonsai, censored, discrim, HSAUR3, lme4, multilevelmod, plsmod, poissonreg, prodlim, rules, sparklyr, survival, and tidymodels. 

There are numerous other "engine" packages that are required. If you use a model that is missing one or more installed packages, parsnip will prompt you to install them. There are some packages that require non-standard installation or rely on external dependencies. We'll describe these next. 

## External Dependencies

Some models available in parsnip use other computational frameworks for computations. There may be some additional downloads for engines using **catboost**, **Spark**, **h2o**, **tensorflow**/**keras**, and **torch**. You can expand the sections below to get basic installation instructions.

<details>

<summary>catboost installation instructions</summary>

catboost is a popular boosting framework. Unfortunately, the R package is not available on CRAN. First, go to [https://github.com/catboost/catboost/releases/]("https://github.com/catboost/catboost/releases/) and search for "`[R-package]`" to find the most recent release. 

The following code can be used to install and test the package (which requires the glue package to be installed): 

::: {.cell layout-align="center"}

```{.r .cell-code}
library(glue)

# Put the current version number in this variable: 
version_number <- "#.#.#" 

template <- "https://github.com/catboost/catboost/releases/download/v{version_number}/catboost-R-darwin-universal2-{version_number}.tgz"

target_url <- glue::glue(template)
target_dest <- tempfile()
download.file(target_url, target_dest)

if (grepl("^mac", .Platform$pkgType)) {
  options <- "--no-staged-install"
} else {
  options <- character(0)
}

inst <- glue::glue("R CMD INSTALL {options} {target_dest}")
system(inst)
```
:::

To test, fit an example model: 

::: {.cell layout-align="center"}

```{.r .cell-code}
library(catboost)

train_pool_path <- system.file("extdata", "adult_train.1000", package = "catboost")
test_pool_path <- system.file("extdata", "adult_test.1000", package = "catboost")
cd_path <- system.file("extdata", "adult.cd", package = "catboost")
train_pool <- catboost.load_pool(train_pool_path, column_description = cd_path)
test_pool <- catboost.load_pool(test_pool_path, column_description = cd_path)
fit_params <- list(
  iterations = 100,
  loss_function = 'Logloss',
  ignored_features = c(4, 9),
  border_count = 32,
  depth = 5,
  learning_rate = 0.03,
  l2_leaf_reg = 3.5,
  train_dir = tempdir())
fit_params
```
:::

</details>

<details>

<summary>Apache Spark installation instructions</summary>

To use [Apache Spark](https://spark.apache.org/) as an engine, we will first install Spark and then need a connection to a cluster. For this article, we will set up and use a single-node Spark cluster running on a laptop.

To install, first install sparklyr:

::: {.cell layout-align="center"}

```{.r .cell-code}
install.packages("sparklyr")
```
:::

and then install the Spark backend. For example, you might use: 

::: {.cell layout-align="center"}

```{.r .cell-code}
library(sparklyr)
spark_install(version = "4.0")
```
:::

Once that is working, you can get ready to fit models using: 

::: {.cell layout-align="center"}

```{.r .cell-code}
library(sparklyr)
sc <- spark_connect("local")
```
:::

</details>

<details>

<summary>h2o installation instructions</summary>

h2o.ai offers a Java-based high-performance computing server for machine learning. This can be run locally or externally. There are general installation instructions at [https://docs.h2o.ai/](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/downloading.html). There is a package on CRAN, but you can also install directly from [h2o](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/downloading.html#install-in-r) via:

::: {.cell layout-align="center"}

```{.r .cell-code}
install.packages(
  "h2o",
  type = "source",
  repos = "http://h2o-release.s3.amazonaws.com/h2o/latest_stable_R"
)
```
:::

After installation is complete, you can start a local server via `h2o::h2o.init()`. 

The tidymodels [agua](https://agua.tidymodels.org/) package contains some helpers and will also need to be installed. You can use its function to start a server too:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(agua)
#> 
#> Attaching package: 'agua'
#> The following object is masked from 'package:workflowsets':
#> 
#>     rank_results
h2o_start()
#> Warning: JAVA not found, H2O may take minutes trying to connect.
```
:::

</details>

<details>

<summary>Tensorflow and Keras installation instructions</summary>

R's tensorflow and keras3 packages call Python directly. To enable this, you'll have to install both components: 

::: {.cell layout-align="center"}

```{.r .cell-code}
install.packages("keras3")
```
:::

Once that is done, use: 

::: {.cell layout-align="center"}

```{.r .cell-code}
keras3::install_keras(backend = "tensorflow")
```
:::

There are other options for installation. See [https://tensorflow.rstudio.com/install/index.html](https://tensorflow.rstudio.com/install/index.html) for more details. 

::: {.cell layout-align="center"}

```{.r .cell-code}
# Assumes you are going to use a virtual environment called 
pve <- grep("tensorflow", reticulate::virtualenv_list(), value = TRUE)
reticulate::use_virtualenv(pve)
```
:::

</details>

<details>

<summary>Torch installation instructions</summary>

R's torch package is the low-level package containing the framework. Once you have installed it, you will get this message the first time you load the package: 

> "Additional software needs to be downloaded and installed for torch to work correctly."

Choosing "Yes" will do the _one-time_ installation. 

</details>

To get started, let's load the tidymodels package: 

::: {.cell layout-align="center"}

```{.r .cell-code}
library(tidymodels)
theme_set(theme_bw() + theme(legend.position = "top"))
```
:::

# Classification Models

## Example data

To demonstrate classification, let's make small training and test sets for a binary outcome. We'll center and scale the data since some models require the same units.

::: {.cell layout-align="center"}

```{.r .cell-code}
set.seed(207)
bin_split <- 
	modeldata::two_class_dat |> 
	rename(class = Class) |> 
	initial_split(prop = 0.994, strata = class)
bin_split
#> <Training/Testing/Total>
#> <785/6/791>

bin_rec <- 
  recipe(class ~ ., data = training(bin_split)) |> 
  step_normalize(all_numeric_predictors()) |> 
  prep()

bin_train <- bake(bin_rec, new_data = NULL)
bin_test <- bake(bin_rec, new_data = testing(bin_split))
```
:::

For models that _only_ work for three or more classes, we'll simulate:

::: {.cell layout-align="center"}

```{.r .cell-code}
set.seed(1752)
mtl_data <-
 sim_multinomial(
    200,
    ~  -0.5 + 0.6 * abs(A),
    ~ ifelse(A > 0 & B > 0, 1.0 + 0.2 * A / B, - 2),
    ~ A + B - A * B
  )

mtl_split <- initial_split(mtl_data, prop = 0.967, strata = class)
mtl_split
#> <Training/Testing/Total>
#> <192/8/200>

# Predictors are in the same units
mtl_train <- training(mtl_split)
mtl_test <- testing(mtl_split)
```
:::

Finally, we have some models that handle hierarchical data, where some rows are statistically correlated with other rows. For these examples, we'll use data from a clinical trial where patients were followed over time. The outcome is binary. The data are in the HSAUR3 package. We'll split these data in a way where all rows for a specific subject are either in the training or test sets: 

::: {.cell layout-align="center"}

```{.r .cell-code}
set.seed(72)
cls_group_split <- 
  HSAUR3::toenail |> 
  group_initial_split(group = patientID)
cls_group_train <- training(cls_group_split)
cls_group_test <- testing(cls_group_split)
```
:::

There are 219 subjects in the training set and 75 in the test set. 

If using the **Apache Spark** engine, we will need to identify the data source and then use it to create the splits. For this article, we will copy the `two_class_dat` and the `mtl_data` data sets into the Spark session.

::: {.cell layout-align="center"}

```{.r .cell-code}
library(sparklyr)
sc <- spark_connect("local")
#> Re-using existing Spark connection to local

tbl_two_class <- copy_to(sc, modeldata::two_class_dat)

tbl_bin <- sdf_random_split(tbl_two_class, training = 0.994, test = 1-0.994, seed = 100)

tbl_sim_mtl <- copy_to(sc, mtl_data)

tbl_mtl <- sdf_random_split(tbl_sim_mtl, training = 0.967, test = 1-0.967, seed = 100)
```
:::

## Models

### Bagged MARS (`bag_mars()`) 

:::{.panel-tabset}

## `earth` 

This engine requires the baguette extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(baguette)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
bag_mars_spec <- bag_mars() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because earth is the default.
  set_mode("classification")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(268)
bag_mars_fit <- bag_mars_spec |> 
  fit(class ~ ., data = bin_train)
#> 
#> Attaching package: 'plotrix'
#> The following object is masked from 'package:scales':
#> 
#>     rescale
#> Registered S3 method overwritten by 'butcher':
#>   method                 from    
#>   as.character.dev_topic generics
bag_mars_fit
#> parsnip model object
#> 
#> Bagged MARS (classification with 11 members)
#> 
#> Variable importance scores include:
#> 
#> # A tibble: 2 × 4
#>   term  value std.error  used
#>   <chr> <dbl>     <dbl> <int>
#> 1 B     100        0       11
#> 2 A      40.4      1.60    11
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(bag_mars_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(bag_mars_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.452       0.548 
#> 2        0.854       0.146 
#> 3        0.455       0.545 
#> 4        0.968       0.0316
#> 5        0.939       0.0610
#> 6        0.872       0.128
```
:::

:::

### Bagged Neural Networks (`bag_mlp()`) 

:::{.panel-tabset}

## `nnet` 

This engine requires the baguette extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(baguette)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
bag_mlp_spec <- bag_mlp() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because nnet is the default.
  set_mode("classification")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(318)
bag_mlp_fit <- bag_mlp_spec |>
  fit(class ~ ., data = bin_train)
bag_mlp_fit
#> parsnip model object
#> 
#> Bagged nnet (classification with 11 members)
#> 
#> Variable importance scores include:
#> 
#> # A tibble: 2 × 4
#>   term  value std.error  used
#>   <chr> <dbl>     <dbl> <int>
#> 1 A      52.1      2.16    11
#> 2 B      47.9      2.16    11
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(bag_mlp_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(bag_mlp_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.439        0.561
#> 2        0.676        0.324
#> 3        0.428        0.572
#> 4        0.727        0.273
#> 5        0.709        0.291
#> 6        0.660        0.340
```
:::

:::

### Bagged Decision Trees (`bag_tree()`) 

:::{.panel-tabset}

## `rpart` 

This engine requires the baguette extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(baguette)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
bag_tree_spec <- bag_tree() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because rpart is the default.
  set_mode("classification")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(985)
bag_tree_fit <- bag_tree_spec |>
  fit(class ~ ., data = bin_train)
bag_tree_fit
#> parsnip model object
#> 
#> Bagged CART (classification with 11 members)
#> 
#> Variable importance scores include:
#> 
#> # A tibble: 2 × 4
#>   term  value std.error  used
#>   <chr> <dbl>     <dbl> <int>
#> 1 B      272.      4.35    11
#> 2 A      237.      5.57    11
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(bag_tree_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(bag_tree_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1       0            1     
#> 2       1            0     
#> 3       0.0909       0.909 
#> 4       1            0     
#> 5       0.727        0.273 
#> 6       0.909        0.0909
```
:::

## `C5.0` 

This engine requires the baguette extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(baguette)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
bag_tree_spec <- bag_tree() |> 
  set_mode("classification") |> 
  set_engine("C5.0")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(937)
bag_tree_fit <- bag_tree_spec |>
  fit(class ~ ., data = bin_train)
bag_tree_fit
#> parsnip model object
#> 
#> Bagged C5.0 (classification with 11 members)
#> 
#> Variable importance scores include:
#> 
#> # A tibble: 2 × 4
#>   term  value std.error  used
#>   <chr> <dbl>     <dbl> <int>
#> 1 B     100        0       11
#> 2 A      48.7      7.33    11
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(bag_tree_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(bag_tree_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.269        0.731
#> 2        0.863        0.137
#> 3        0.259        0.741
#> 4        0.897        0.103
#> 5        0.897        0.103
#> 6        0.870        0.130
```
:::

:::

### Bayesian Additive Regression Trees (`bart()`)

:::{.panel-tabset}

## `dbarts`

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
bart_spec <- bart() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because dbarts is the default.
  set_mode("classification")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(217)
bart_fit <- bart_spec |>
  fit(class ~ ., data = bin_train)
bart_fit
#> parsnip model object
#> 
#> 
#> Call:
#> `NULL`()
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(bart_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(bart_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.439        0.561
#> 2        0.734        0.266
#> 3        0.34         0.66 
#> 4        0.957        0.043
#> 5        0.931        0.069
#> 6        0.782        0.218
predict(bart_fit, type = "conf_int", new_data = bin_test)
#> # A tibble: 6 × 4
#>   .pred_lower_Class1 .pred_lower_Class2 .pred_upper_Class1 .pred_upper_Class2
#>                <dbl>              <dbl>              <dbl>              <dbl>
#> 1              0.815            0.00280              0.997              0.185
#> 2              0.781            0.0223               0.978              0.219
#> 3              0.558            0.0702               0.930              0.442
#> 4              0.540            0.105                0.895              0.460
#> 5              0.239            0.345                0.655              0.761
#> 6              0.195            0.469                0.531              0.805
predict(bart_fit, type = "pred_int", new_data = bin_test)
#> # A tibble: 6 × 4
#>   .pred_lower_Class1 .pred_lower_Class2 .pred_upper_Class1 .pred_upper_Class2
#>                <dbl>              <dbl>              <dbl>              <dbl>
#> 1                  0                  0                  1                  1
#> 2                  0                  0                  1                  1
#> 3                  0                  0                  1                  1
#> 4                  0                  0                  1                  1
#> 5                  0                  0                  1                  1
#> 6                  0                  0                  1                  1
```
:::

:::

### Boosted Decision Trees (`boost_tree()`)

:::{.panel-tabset}

## `xgboost`

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
boost_tree_spec <- boost_tree() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because xgboost is the default.
  set_mode("classification")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(738)
boost_tree_fit <- boost_tree_spec |>
  fit(class ~ ., data = bin_train)
boost_tree_fit
#> parsnip model object
#> 
#> ##### xgb.Booster
#> call:
#>   xgboost::xgb.train(params = list(eta = 0.3, max_depth = 6, gamma = 0, 
#>     colsample_bytree = 1, colsample_bynode = 1, min_child_weight = 1, 
#>     subsample = 1, nthread = 1, objective = "binary:logistic"), 
#>     data = x$data, nrounds = 15, evals = x$watchlist, verbose = 0)
#> # of features: 2 
#> # of rounds:  15 
#> callbacks:
#>    evaluation_log 
#> evaluation_log:
#>   iter training_logloss
#>  <num>            <num>
#>      1        0.5486970
#>      2        0.4698863
#>    ---              ---
#>     14        0.2646550
#>     15        0.2616105
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(boost_tree_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(boost_tree_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.214       0.786 
#> 2        0.766       0.234 
#> 3        0.289       0.711 
#> 4        0.962       0.0377
#> 5        0.831       0.169 
#> 6        0.945       0.0552
```
:::

## `C5.0` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
boost_tree_spec <- boost_tree() |> 
  set_mode("classification") |> 
  set_engine("C5.0")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(984)
boost_tree_fit <- boost_tree_spec |>
  fit(class ~ ., data = bin_train)
boost_tree_fit
#> parsnip model object
#> 
#> 
#> Call:
#> C5.0.default(x = x, y = y, trials = 15, control = C50::C5.0Control(minCases
#>  = 2, sample = 0))
#> 
#> Classification Tree
#> Number of samples: 785 
#> Number of predictors: 2 
#> 
#> Number of boosting iterations: 15 requested;  7 used due to early stopping
#> Average tree size: 3.1 
#> 
#> Non-standard options: attempt to group attributes
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(boost_tree_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(boost_tree_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.307        0.693
#> 2        0.756        0.244
#> 3        0.281        0.719
#> 4        1            0    
#> 5        1            0    
#> 6        0.626        0.374
```
:::

## `catboost` 

This engine requires the bonsai extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(bonsai)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
boost_tree_spec <- boost_tree() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("classification") |>
  set_engine("catboost")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(644)
boost_tree_fit <- boost_tree_spec |>
  fit(class ~ ., data = bin_train)
boost_tree_fit
#> parsnip model object
#> 
#> CatBoost model (1000 trees)
#> Loss function: Logloss
#> Fit to 2 feature(s)
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(boost_tree_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(boost_tree_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.291      0.709  
#> 2        0.836      0.164  
#> 3        0.344      0.656  
#> 4        0.998      0.00245
#> 5        0.864      0.136  
#> 6        0.902      0.0983
```
:::

## `h2o` 

This engine requires the agua extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(agua)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
boost_tree_spec <- boost_tree() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("classification") |>
  set_engine("h2o_gbm")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(186)
boost_tree_fit <- boost_tree_spec |>
  fit(class ~ ., data = bin_train)
boost_tree_fit
#> parsnip model object
#> 
#> Model Details:
#> ==============
#> 
#> H2OBinomialModel: gbm
#> Model ID:  GBM_model_R_1770287512312_5613 
#> Model Summary: 
#>   number_of_trees number_of_internal_trees model_size_in_bytes min_depth
#> 1              50                       50               25380         6
#>   max_depth mean_depth min_leaves max_leaves mean_leaves
#> 1         6    6.00000         21         55    35.70000
#> 
#> 
#> H2OBinomialMetrics: gbm
#> ** Reported on training data. **
#> 
#> MSE:  0.007948832
#> RMSE:  0.08915622
#> LogLoss:  0.05942305
#> Mean Per-Class Error:  0
#> AUC:  1
#> AUCPR:  1
#> Gini:  1
#> R^2:  0.9678452
#> AIC:  NaN
#> 
#> Confusion Matrix (vertical: actual; across: predicted) for F1-optimal threshold:
#>        Class1 Class2    Error    Rate
#> Class1    434      0 0.000000  =0/434
#> Class2      0    351 0.000000  =0/351
#> Totals    434    351 0.000000  =0/785
#> 
#> Maximum Metrics: Maximum metrics at their respective thresholds
#>                         metric threshold      value idx
#> 1                       max f1  0.598690   1.000000 200
#> 2                       max f2  0.598690   1.000000 200
#> 3                 max f0point5  0.598690   1.000000 200
#> 4                 max accuracy  0.598690   1.000000 200
#> 5                max precision  0.998192   1.000000   0
#> 6                   max recall  0.598690   1.000000 200
#> 7              max specificity  0.998192   1.000000   0
#> 8             max absolute_mcc  0.598690   1.000000 200
#> 9   max min_per_class_accuracy  0.598690   1.000000 200
#> 10 max mean_per_class_accuracy  0.598690   1.000000 200
#> 11                     max tns  0.998192 434.000000   0
#> 12                     max fns  0.998192 349.000000   0
#> 13                     max fps  0.000831 434.000000 399
#> 14                     max tps  0.598690 351.000000 200
#> 15                     max tnr  0.998192   1.000000   0
#> 16                     max fnr  0.998192   0.994302   0
#> 17                     max fpr  0.000831   1.000000 399
#> 18                     max tpr  0.598690   1.000000 200
#> 
#> Gains/Lift Table: Extract with `h2o.gainsLift(<model>, <data>)` or `h2o.gainsLift(<model>, valid=<T/F>, xval=<T/F>)`
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(boost_tree_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(boost_tree_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1       0.0496      0.950  
#> 2       0.905       0.0953 
#> 3       0.0738      0.926  
#> 4       0.997       0.00273
#> 5       0.979       0.0206 
#> 6       0.878       0.122
```
:::

## `h2o_gbm` 

This engine requires the agua extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(agua)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
boost_tree_spec <- boost_tree() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("classification") |>
  set_engine("h2o_gbm")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(724)
boost_tree_fit <- boost_tree_spec |>
  fit(class ~ ., data = bin_train)
boost_tree_fit
#> parsnip model object
#> 
#> Model Details:
#> ==============
#> 
#> H2OBinomialModel: gbm
#> Model ID:  GBM_model_R_1770287512312_5665 
#> Model Summary: 
#>   number_of_trees number_of_internal_trees model_size_in_bytes min_depth
#> 1              50                       50               25379         6
#>   max_depth mean_depth min_leaves max_leaves mean_leaves
#> 1         6    6.00000         21         55    35.70000
#> 
#> 
#> H2OBinomialMetrics: gbm
#> ** Reported on training data. **
#> 
#> MSE:  0.007948832
#> RMSE:  0.08915622
#> LogLoss:  0.05942305
#> Mean Per-Class Error:  0
#> AUC:  1
#> AUCPR:  1
#> Gini:  1
#> R^2:  0.9678452
#> AIC:  NaN
#> 
#> Confusion Matrix (vertical: actual; across: predicted) for F1-optimal threshold:
#>        Class1 Class2    Error    Rate
#> Class1    434      0 0.000000  =0/434
#> Class2      0    351 0.000000  =0/351
#> Totals    434    351 0.000000  =0/785
#> 
#> Maximum Metrics: Maximum metrics at their respective thresholds
#>                         metric threshold      value idx
#> 1                       max f1  0.598690   1.000000 200
#> 2                       max f2  0.598690   1.000000 200
#> 3                 max f0point5  0.598690   1.000000 200
#> 4                 max accuracy  0.598690   1.000000 200
#> 5                max precision  0.998192   1.000000   0
#> 6                   max recall  0.598690   1.000000 200
#> 7              max specificity  0.998192   1.000000   0
#> 8             max absolute_mcc  0.598690   1.000000 200
#> 9   max min_per_class_accuracy  0.598690   1.000000 200
#> 10 max mean_per_class_accuracy  0.598690   1.000000 200
#> 11                     max tns  0.998192 434.000000   0
#> 12                     max fns  0.998192 349.000000   0
#> 13                     max fps  0.000831 434.000000 399
#> 14                     max tps  0.598690 351.000000 200
#> 15                     max tnr  0.998192   1.000000   0
#> 16                     max fnr  0.998192   0.994302   0
#> 17                     max fpr  0.000831   1.000000 399
#> 18                     max tpr  0.598690   1.000000 200
#> 
#> Gains/Lift Table: Extract with `h2o.gainsLift(<model>, <data>)` or `h2o.gainsLift(<model>, valid=<T/F>, xval=<T/F>)`
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(boost_tree_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(boost_tree_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1       0.0496      0.950  
#> 2       0.905       0.0953 
#> 3       0.0738      0.926  
#> 4       0.997       0.00273
#> 5       0.979       0.0206 
#> 6       0.878       0.122
```
:::

## `lightgbm` 

This engine requires the bonsai extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(bonsai)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
boost_tree_spec <- boost_tree() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("classification") |>
  set_engine("lightgbm")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(906)
boost_tree_fit <- boost_tree_spec |>
  fit(class ~ ., data = bin_train)
boost_tree_fit
#> parsnip model object
#> 
#> LightGBM Model (100 trees)
#> Objective: binary
#> Fitted to dataset with 2 columns
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(boost_tree_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(boost_tree_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.147       0.853 
#> 2        0.930       0.0699
#> 3        0.237       0.763 
#> 4        0.990       0.0101
#> 5        0.929       0.0714
#> 6        0.956       0.0445
```
:::

## `spark` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
boost_tree_spec <- boost_tree() |> 
  set_mode("classification") |> 
  set_engine("spark")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(285)
boost_tree_fit <- boost_tree_spec |>
  fit(Class ~ ., data = tbl_bin$training)
boost_tree_fit
#> parsnip model object
#> 
#> Formula: Class ~ .
#> 
#> GBTClassificationModel: uid = gradient_boosted_trees__1c57358a_ec7a_4c0c_8911_2c8d306fa909, numTrees=20, numClasses=2, numFeatures=2
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(boost_tree_fit, type = "class", new_data = tbl_bin$test)
#> # Source:   SQL [?? x 1]
#> # Database: spark_connection
#>   pred_class
#>   <chr>     
#> 1 Class2    
#> 2 Class2    
#> 3 Class1    
#> 4 Class2    
#> 5 Class2    
#> 6 Class1    
#> 7 Class2
predict(boost_tree_fit, type = "prob", new_data = tbl_bin$test)
#> # Source:   SQL [?? x 2]
#> # Database: spark_connection
#>   pred_Class1 pred_Class2
#>         <dbl>       <dbl>
#> 1      0.307       0.693 
#> 2      0.292       0.708 
#> 3      0.856       0.144 
#> 4      0.192       0.808 
#> 5      0.332       0.668 
#> 6      0.952       0.0476
#> 7      0.0865      0.914
```
:::

:::

### C5 Rules (`C5_rules()`)

:::{.panel-tabset}

## `C5.0` 

This engine requires the rules extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(rules)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
# This model only works with a single mode, so we don't need to set the mode.
# We don't need to set the engine because C5.0 is the default.
C5_rules_spec <- C5_rules()
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(93)
C5_rules_fit <- C5_rules_spec |>
  fit(class ~ ., data = bin_train)
C5_rules_fit
#> parsnip model object
#> 
#> 
#> Call:
#> C5.0.default(x = x, y = y, trials = trials, rules = TRUE, control
#>  = C50::C5.0Control(minCases = minCases, seed = sample.int(10^5,
#>  1), earlyStopping = FALSE))
#> 
#> Rule-Based Model
#> Number of samples: 785 
#> Number of predictors: 2 
#> 
#> Number of Rules: 4 
#> 
#> Non-standard options: attempt to group attributes
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(C5_rules_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class1     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(C5_rules_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1            1            0
#> 2            1            0
#> 3            0            1
#> 4            1            0
#> 5            1            0
#> 6            1            0
```
:::

:::

### Decision Tree (`decision_tree()`)

:::{.panel-tabset}

## `rpart`

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
decision_tree_spec <- decision_tree() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because rpart is the default.
  set_mode("classification")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
decision_tree_fit <- decision_tree_spec |>
  fit(class ~ ., data = bin_train)
decision_tree_fit
#> parsnip model object
#> 
#> n= 785 
#> 
#> node), split, n, loss, yval, (yprob)
#>       * denotes terminal node
#> 
#>  1) root 785 351 Class1 (0.5528662 0.4471338)  
#>    2) B< -0.06526451 399  61 Class1 (0.8471178 0.1528822) *
#>    3) B>=-0.06526451 386  96 Class2 (0.2487047 0.7512953)  
#>      6) B< 0.7339337 194  72 Class2 (0.3711340 0.6288660)  
#>       12) A>=0.6073948 49  13 Class1 (0.7346939 0.2653061) *
#>       13) A< 0.6073948 145  36 Class2 (0.2482759 0.7517241) *
#>      7) B>=0.7339337 192  24 Class2 (0.1250000 0.8750000) *
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(decision_tree_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class1     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(decision_tree_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.735        0.265
#> 2        0.847        0.153
#> 3        0.248        0.752
#> 4        0.847        0.153
#> 5        0.847        0.153
#> 6        0.847        0.153
```
:::

## `C5.0` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
decision_tree_spec <- decision_tree() |> 
  set_mode("classification") |> 
  set_engine("C5.0")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
decision_tree_fit <- decision_tree_spec |>
  fit(class ~ ., data = bin_train)
decision_tree_fit
#> parsnip model object
#> 
#> 
#> Call:
#> C5.0.default(x = x, y = y, trials = 1, control = C50::C5.0Control(minCases =
#>  2, sample = 0))
#> 
#> Classification Tree
#> Number of samples: 785 
#> Number of predictors: 2 
#> 
#> Tree size: 4 
#> 
#> Non-standard options: attempt to group attributes
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(decision_tree_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class1     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(decision_tree_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.732        0.268
#> 2        0.846        0.154
#> 3        0.236        0.764
#> 4        0.846        0.154
#> 5        0.846        0.154
#> 6        0.846        0.154
```
:::

## `partykit` 

This engine requires the bonsai extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(bonsai)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
decision_tree_spec <- decision_tree() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("classification") |>
  set_engine("partykit")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
decision_tree_fit <- decision_tree_spec |>
  fit(class ~ ., data = bin_train)
decision_tree_fit
#> parsnip model object
#> 
#> 
#> Model formula:
#> class ~ A + B
#> 
#> Fitted party:
#> [1] root
#> |   [2] B <= -0.06906
#> |   |   [3] B <= -0.50486: Class1 (n = 291, err = 8.2%)
#> |   |   [4] B > -0.50486
#> |   |   |   [5] A <= -0.07243: Class1 (n = 77, err = 45.5%)
#> |   |   |   [6] A > -0.07243: Class1 (n = 31, err = 6.5%)
#> |   [7] B > -0.06906
#> |   |   [8] B <= 0.72938
#> |   |   |   [9] A <= 0.60196: Class2 (n = 145, err = 24.8%)
#> |   |   |   [10] A > 0.60196
#> |   |   |   |   [11] B <= 0.44701: Class1 (n = 23, err = 4.3%)
#> |   |   |   |   [12] B > 0.44701: Class1 (n = 26, err = 46.2%)
#> |   |   [13] B > 0.72938: Class2 (n = 192, err = 12.5%)
#> 
#> Number of inner nodes:    6
#> Number of terminal nodes: 7
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(decision_tree_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class1     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(decision_tree_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.538       0.462 
#> 2        0.935       0.0645
#> 3        0.248       0.752 
#> 4        0.918       0.0825
#> 5        0.918       0.0825
#> 6        0.935       0.0645
```
:::

## `spark` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
decision_tree_spec <- decision_tree() |>
  set_mode("classification") |> 
  set_engine("spark")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
decision_tree_fit <- decision_tree_spec |>
  fit(Class ~ ., data = tbl_bin$training)
decision_tree_fit
#> parsnip model object
#> 
#> Formula: Class ~ .
#> 
#> DecisionTreeClassificationModel: uid=decision_tree_classifier__a4e63506_aa0e_40d3_9ba8_a6534c10629f, depth=5, numNodes=43, numClasses=2, numFeatures=2
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(decision_tree_fit, type = "class", new_data = tbl_bin$test)
#> # Source:   SQL [?? x 1]
#> # Database: spark_connection
#>   pred_class
#>   <chr>     
#> 1 Class2    
#> 2 Class2    
#> 3 Class1    
#> 4 Class2    
#> 5 Class2    
#> 6 Class1    
#> 7 Class2
predict(decision_tree_fit, type = "prob", new_data = tbl_bin$test)
#> # Source:   SQL [?? x 2]
#> # Database: spark_connection
#>   pred_Class1 pred_Class2
#>         <dbl>       <dbl>
#> 1      0.260       0.740 
#> 2      0.260       0.740 
#> 3      0.860       0.140 
#> 4      0.260       0.740 
#> 5      0.260       0.740 
#> 6      0.923       0.0769
#> 7      0.0709      0.929
```
:::

:::

### Flexible Discriminant Analysis (`discrim_flexible()`)

:::{.panel-tabset}

## `earth` 

This engine requires the discrim extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(discrim)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
# This model only works with a single mode, so we don't need to set the mode.
# We don't need to set the engine because earth is the default.
discrim_flexible_spec <- discrim_flexible()
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
discrim_flexible_fit <- discrim_flexible_spec |>
  fit(class ~ ., data = bin_train)
discrim_flexible_fit
#> parsnip model object
#> 
#> Call:
#> mda::fda(formula = class ~ ., data = data, method = earth::earth)
#> 
#> Dimension: 1 
#> 
#> Percent Between-Group Variance Explained:
#>  v1 
#> 100 
#> 
#> Training Misclassification Error: 0.1707 ( N = 785 )
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(discrim_flexible_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(discrim_flexible_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.339       0.661 
#> 2        0.848       0.152 
#> 3        0.342       0.658 
#> 4        0.964       0.0360
#> 5        0.964       0.0360
#> 6        0.875       0.125
```
:::

:::

### Linear Discriminant Analysis (`discrim_linear()`)

:::{.panel-tabset}

## `MASS` 

This engine requires the discrim extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(discrim)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
# This model only works with a single mode, so we don't need to set the mode.
# We don't need to set the engine because MASS is the default.
discrim_linear_spec <- discrim_linear()
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
discrim_linear_fit <- discrim_linear_spec |>
  fit(class ~ ., data = bin_train)
discrim_linear_fit
#> parsnip model object
#> 
#> Call:
#> lda(class ~ ., data = data)
#> 
#> Prior probabilities of groups:
#>    Class1    Class2 
#> 0.5528662 0.4471338 
#> 
#> Group means:
#>                 A          B
#> Class1 -0.2982900 -0.5573140
#> Class2  0.3688258  0.6891006
#> 
#> Coefficients of linear discriminants:
#>          LD1
#> A -0.6068479
#> B  1.7079953
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(discrim_linear_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class1     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(discrim_linear_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.369       0.631 
#> 2        0.868       0.132 
#> 3        0.541       0.459 
#> 4        0.984       0.0158
#> 5        0.928       0.0718
#> 6        0.854       0.146
```
:::

## `mda` 

This engine requires the discrim extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(discrim)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
discrim_linear_spec <- discrim_linear() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("mda")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
discrim_linear_fit <- discrim_linear_spec |>
  fit(class ~ ., data = bin_train)
discrim_linear_fit
#> parsnip model object
#> 
#> Call:
#> mda::fda(formula = class ~ ., data = data, method = mda::gen.ridge, 
#>     keep.fitted = FALSE)
#> 
#> Dimension: 1 
#> 
#> Percent Between-Group Variance Explained:
#>  v1 
#> 100 
#> 
#> Degrees of Freedom (per dimension): 1.99423 
#> 
#> Training Misclassification Error: 0.17707 ( N = 785 )
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(discrim_linear_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class1     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(discrim_linear_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.368       0.632 
#> 2        0.867       0.133 
#> 3        0.542       0.458 
#> 4        0.984       0.0158
#> 5        0.928       0.0718
#> 6        0.853       0.147
```
:::

## `sda` 

This engine requires the discrim extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(discrim)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
discrim_linear_spec <- discrim_linear() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("sda")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
discrim_linear_fit <- discrim_linear_spec |>
  fit(class ~ ., data = bin_train)
discrim_linear_fit
#> parsnip model object
#> 
#> $regularization
#>       lambda   lambda.var lambda.freqs 
#>  0.003136201  0.067551534  0.112819609 
#> 
#> $freqs
#>    Class1    Class2 
#> 0.5469019 0.4530981 
#> 
#> $alpha
#>     Class1     Class2 
#> -0.8934125 -1.2349286 
#> 
#> $beta
#>                 A         B
#> Class1  0.4565325 -1.298858
#> Class2 -0.5510473  1.567757
#> attr(,"class")
#> [1] "shrinkage"
#> 
#> attr(,"class")
#> [1] "sda"
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(discrim_linear_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class1     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(discrim_linear_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.366       0.634 
#> 2        0.860       0.140 
#> 3        0.536       0.464 
#> 4        0.982       0.0176
#> 5        0.923       0.0768
#> 6        0.845       0.155
```
:::

## `sparsediscrim` 

This engine requires the discrim extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(discrim)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
discrim_linear_spec <- discrim_linear() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("sparsediscrim")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
discrim_linear_fit <- discrim_linear_spec |>
  fit(class ~ ., data = bin_train)
discrim_linear_fit
#> parsnip model object
#> 
#> Diagonal LDA
#> 
#> Sample Size: 785 
#> Number of Features: 2 
#> 
#> Classes and Prior Probabilities:
#>   Class1 (55.29%), Class2 (44.71%)
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(discrim_linear_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(discrim_linear_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.182      0.818  
#> 2        0.755      0.245  
#> 3        0.552      0.448  
#> 4        0.996      0.00372
#> 5        0.973      0.0274 
#> 6        0.629      0.371
```
:::

:::

### Quandratic Discriminant Analysis (`discrim_quad()`)

:::{.panel-tabset}

## `MASS` 

This engine requires the discrim extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(discrim)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
discrim_quad_spec <- discrim_quad()
  # This model only works with a single mode, so we don't need to set the mode.
  # We don't need to set the engine because MASS is the default.
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
discrim_quad_fit <- discrim_quad_spec |>
  fit(class ~ ., data = bin_train)
discrim_quad_fit
#> parsnip model object
#> 
#> Call:
#> qda(class ~ ., data = data)
#> 
#> Prior probabilities of groups:
#>    Class1    Class2 
#> 0.5528662 0.4471338 
#> 
#> Group means:
#>                 A          B
#> Class1 -0.2982900 -0.5573140
#> Class2  0.3688258  0.6891006
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(discrim_quad_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class1     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(discrim_quad_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.340       0.660 
#> 2        0.884       0.116 
#> 3        0.500       0.500 
#> 4        0.965       0.0349
#> 5        0.895       0.105 
#> 6        0.895       0.105
```
:::

## `sparsediscrim` 

This engine requires the discrim extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(discrim)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
discrim_quad_spec <- discrim_quad() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("sparsediscrim")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
discrim_quad_fit <- discrim_quad_spec |>
  fit(class ~ ., data = bin_train)
discrim_quad_fit
#> parsnip model object
#> 
#> Diagonal QDA
#> 
#> Sample Size: 785 
#> Number of Features: 2 
#> 
#> Classes and Prior Probabilities:
#>   Class1 (55.29%), Class2 (44.71%)
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(discrim_quad_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(discrim_quad_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.180      0.820  
#> 2        0.750      0.250  
#> 3        0.556      0.444  
#> 4        0.994      0.00634
#> 5        0.967      0.0328 
#> 6        0.630      0.370
```
:::

:::

### Regularized Discriminant Analysis (`discrim_regularized()`)

:::{.panel-tabset}

## `klaR` 

This engine requires the discrim extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(discrim)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
# This model only works with a single mode, so we don't need to set the mode.
# We don't need to set the engine because klaR is the default.
discrim_regularized_spec <- discrim_regularized()
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
discrim_regularized_fit <- discrim_regularized_spec |>
  fit(class ~ ., data = bin_train)
discrim_regularized_fit
#> parsnip model object
#> 
#> Call: 
#> rda(formula = class ~ ., data = data)
#> 
#> Regularization parameters: 
#>        gamma       lambda 
#> 3.348721e-05 3.288193e-04 
#> 
#> Prior probabilities of groups: 
#>    Class1    Class2 
#> 0.5528662 0.4471338 
#> 
#> Misclassification rate: 
#>        apparent: 17.707 %
#> cross-validated: 17.566 %
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(discrim_regularized_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class1     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(discrim_regularized_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.340       0.660 
#> 2        0.884       0.116 
#> 3        0.501       0.499 
#> 4        0.965       0.0349
#> 5        0.895       0.105 
#> 6        0.895       0.105
```
:::

:::

### Generalized Additive Models (`gen_additive_mod()`)

:::{.panel-tabset}

## `mgcv`

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
gen_additive_mod_spec <- gen_additive_mod() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because mgcv is the default.
  set_mode("classification")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
gen_additive_mod_fit <- 
  gen_additive_mod_spec |> 
  fit(class ~ s(A) + s(B), data = bin_train)
gen_additive_mod_fit
#> parsnip model object
#> 
#> 
#> Family: binomial 
#> Link function: logit 
#> 
#> Formula:
#> class ~ s(A) + s(B)
#> 
#> Estimated degrees of freedom:
#> 2.76 4.22  total = 7.98 
#> 
#> UBRE score: -0.153537
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(gen_additive_mod_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(gen_additive_mod_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.400       0.600 
#> 2        0.826       0.174 
#> 3        0.454       0.546 
#> 4        0.975       0.0250
#> 5        0.929       0.0711
#> 6        0.829       0.171
predict(gen_additive_mod_fit, type = "conf_int", new_data = bin_test)
#> # A tibble: 6 × 4
#>   .pred_lower_Class1 .pred_upper_Class1 .pred_lower_Class2 .pred_upper_Class2
#>            <dbl[1d]>          <dbl[1d]>          <dbl[1d]>          <dbl[1d]>
#> 1              0.304              0.504            0.496                0.696
#> 2              0.739              0.889            0.111                0.261
#> 3              0.364              0.546            0.454                0.636
#> 4              0.846              0.996            0.00358              0.154
#> 5              0.881              0.958            0.0416               0.119
#> 6              0.735              0.894            0.106                0.265
```
:::

:::

### Logistic Regression (`logistic_reg()`)

:::{.panel-tabset}

## `glm` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
logistic_reg_spec <- logistic_reg()
  # This model only works with a single mode, so we don't need to set the mode.
  # We don't need to set the engine because glm is the default.
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
logistic_reg_fit <- logistic_reg_spec |>
  fit(class ~ ., data = bin_train)
logistic_reg_fit
#> parsnip model object
#> 
#> 
#> Call:  stats::glm(formula = class ~ ., family = stats::binomial, data = data)
#> 
#> Coefficients:
#> (Intercept)            A            B  
#>     -0.3563      -1.1250       2.8154  
#> 
#> Degrees of Freedom: 784 Total (i.e. Null);  782 Residual
#> Null Deviance:	    1079 
#> Residual Deviance: 666.9 	AIC: 672.9
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(logistic_reg_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class1     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(logistic_reg_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.400       0.600 
#> 2        0.862       0.138 
#> 3        0.541       0.459 
#> 4        0.977       0.0234
#> 5        0.909       0.0905
#> 6        0.853       0.147
predict(logistic_reg_fit, type = "conf_int", new_data = bin_test)
#> # A tibble: 6 × 4
#>   .pred_lower_Class1 .pred_upper_Class1 .pred_lower_Class2 .pred_upper_Class2
#>                <dbl>              <dbl>              <dbl>              <dbl>
#> 1              0.339              0.465             0.535              0.661 
#> 2              0.816              0.897             0.103              0.184 
#> 3              0.493              0.588             0.412              0.507 
#> 4              0.960              0.986             0.0137             0.0395
#> 5              0.875              0.935             0.0647             0.125 
#> 6              0.800              0.894             0.106              0.200
```
:::

## `brulee` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
logistic_reg_spec <- logistic_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("brulee")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(466)
logistic_reg_fit <- logistic_reg_spec |>
  fit(class ~ ., data = bin_train)
logistic_reg_fit
#> parsnip model object
#> 
#> Logistic regression
#> 
#> 785 samples, 2 features, 2 classes 
#> class weights Class1=1, Class2=1 
#> weight decay: 0.001 
#> batch size: 707 
#> validation loss after 1 epoch: 0.283
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(logistic_reg_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class1     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(logistic_reg_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.412       0.588 
#> 2        0.854       0.146 
#> 3        0.537       0.463 
#> 4        0.971       0.0294
#> 5        0.896       0.104 
#> 6        0.848       0.152
```
:::

## `gee` 

This engine requires the multilevelmod extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(multilevelmod)

```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
logistic_reg_spec <- logistic_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("gee")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
logistic_reg_fit <- 
  logistic_reg_spec |> 
  fit(outcome ~ treatment * visit + id_var(patientID), data = cls_group_train)
#> Beginning Cgee S-function, @(#) geeformula.q 4.13 98/01/27
#> running glm to get initial regression estimate
logistic_reg_fit
#> parsnip model object
#> 
#> 
#>  GEE:  GENERALIZED LINEAR MODELS FOR DEPENDENT DATA
#>  gee S-function, version 4.13 modified 98/01/27 (1998) 
#> 
#> Model:
#>  Link:                      Logit 
#>  Variance to Mean Relation: Binomial 
#>  Correlation Structure:     Independent 
#> 
#> Call:
#> gee::gee(formula = outcome ~ treatment + visit, id = data$patientID, 
#>     data = data, family = binomial)
#> 
#> Number of observations :  1433 
#> 
#> Maximum cluster size   :  7 
#> 
#> 
#> Coefficients:
#>          (Intercept) treatmentterbinafine                visit 
#>          -0.06853546          -0.25700680          -0.35646522 
#> 
#> Estimated Scale Parameter:  0.9903994
#> Number of Iterations:  1
#> 
#> Working Correlation[1:4,1:4]
#>      [,1] [,2] [,3] [,4]
#> [1,]    1    0    0    0
#> [2,]    0    1    0    0
#> [3,]    0    0    1    0
#> [4,]    0    0    0    1
#> 
#> 
#> Returned Error Value:
#> [1] 0
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(logistic_reg_fit, type = "class", new_data = cls_group_test)
#> # A tibble: 475 × 1
#>    .pred_class 
#>    <fct>       
#>  1 none or mild
#>  2 none or mild
#>  3 none or mild
#>  4 none or mild
#>  5 none or mild
#>  6 none or mild
#>  7 none or mild
#>  8 none or mild
#>  9 none or mild
#> 10 none or mild
#> # ℹ 465 more rows
predict(logistic_reg_fit, type = "prob", new_data = cls_group_test)
#> # A tibble: 475 × 2
#>    `.pred_none or mild` `.pred_moderate or severe`
#>                   <dbl>                      <dbl>
#>  1                0.664                     0.336 
#>  2                0.739                     0.261 
#>  3                0.801                     0.199 
#>  4                0.852                     0.148 
#>  5                0.892                     0.108 
#>  6                0.922                     0.0784
#>  7                0.944                     0.0562
#>  8                0.605                     0.395 
#>  9                0.686                     0.314 
#> 10                0.757                     0.243 
#> # ℹ 465 more rows
```
:::

## `glmer` 

This engine requires the multilevelmod extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(multilevelmod)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
logistic_reg_spec <- logistic_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("glmer")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
logistic_reg_fit <- 
  logistic_reg_spec |> 
  fit(outcome ~ treatment * visit + (1 | patientID), data = cls_group_train)
logistic_reg_fit
#> parsnip model object
#> 
#> Generalized linear mixed model fit by maximum likelihood (Laplace
#>   Approximation) [glmerMod]
#>  Family: binomial  ( logit )
#> Formula: outcome ~ treatment * visit + (1 | patientID)
#>    Data: data
#>       AIC       BIC    logLik -2*log(L)  df.resid 
#>  863.8271  890.1647 -426.9135  853.8271      1428 
#> Random effects:
#>  Groups    Name        Std.Dev.
#>  patientID (Intercept) 8.35    
#> Number of obs: 1433, groups:  patientID, 219
#> Fixed Effects:
#>                (Intercept)        treatmentterbinafine  
#>                  -4.574209                   -0.511919  
#>                      visit  treatmentterbinafine:visit  
#>                  -0.987246                   -0.001121
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(logistic_reg_fit, type = "class", new_data = cls_group_test)
#> # A tibble: 475 × 1
#>    .pred_class 
#>    <fct>       
#>  1 none or mild
#>  2 none or mild
#>  3 none or mild
#>  4 none or mild
#>  5 none or mild
#>  6 none or mild
#>  7 none or mild
#>  8 none or mild
#>  9 none or mild
#> 10 none or mild
#> # ℹ 465 more rows
predict(logistic_reg_fit, type = "prob", new_data = cls_group_test)
#> # A tibble: 475 × 2
#>    `.pred_none or mild` `.pred_moderate or severe`
#>                   <dbl>                      <dbl>
#>  1                0.998                 0.00230   
#>  2                0.999                 0.000856  
#>  3                1.000                 0.000319  
#>  4                1.000                 0.000119  
#>  5                1.000                 0.0000441 
#>  6                1.000                 0.0000164 
#>  7                1.000                 0.00000612
#>  8                0.996                 0.00383   
#>  9                0.999                 0.00143   
#> 10                0.999                 0.000533  
#> # ℹ 465 more rows
```
:::

## `glmnet` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
logistic_reg_spec <- logistic_reg(penalty = 0.01) |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("glmnet")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
logistic_reg_fit <- logistic_reg_spec |>
  fit(class ~ ., data = bin_train)
logistic_reg_fit
#> parsnip model object
#> 
#> 
#> Call:  glmnet::glmnet(x = maybe_matrix(x), y = y, family = "binomial") 
#> 
#>    Df  %Dev   Lambda
#> 1   0  0.00 0.308300
#> 2   1  4.75 0.280900
#> 3   1  8.73 0.256000
#> 4   1 12.10 0.233200
#> 5   1 14.99 0.212500
#> 6   1 17.46 0.193600
#> 7   1 19.60 0.176400
#> 8   1 21.45 0.160800
#> 9   1 23.05 0.146500
#> 10  1 24.44 0.133500
#> 11  1 25.65 0.121600
#> 12  1 26.70 0.110800
#> 13  1 27.61 0.101000
#> 14  1 28.40 0.091990
#> 15  1 29.08 0.083820
#> 16  1 29.68 0.076370
#> 17  1 30.19 0.069590
#> 18  1 30.63 0.063410
#> 19  1 31.00 0.057770
#> 20  1 31.33 0.052640
#> 21  1 31.61 0.047960
#> 22  1 31.85 0.043700
#> 23  1 32.05 0.039820
#> 24  2 32.62 0.036280
#> 25  2 33.41 0.033060
#> 26  2 34.10 0.030120
#> 27  2 34.68 0.027450
#> 28  2 35.19 0.025010
#> 29  2 35.63 0.022790
#> 30  2 36.01 0.020760
#> 31  2 36.33 0.018920
#> 32  2 36.62 0.017240
#> 33  2 36.86 0.015710
#> 34  2 37.06 0.014310
#> 35  2 37.24 0.013040
#> 36  2 37.39 0.011880
#> 37  2 37.52 0.010830
#> 38  2 37.63 0.009864
#> 39  2 37.72 0.008988
#> 40  2 37.80 0.008189
#> 41  2 37.86 0.007462
#> 42  2 37.92 0.006799
#> 43  2 37.97 0.006195
#> 44  2 38.01 0.005644
#> 45  2 38.04 0.005143
#> 46  2 38.07 0.004686
#> 47  2 38.10 0.004270
#> 48  2 38.12 0.003891
#> 49  2 38.13 0.003545
#> 50  2 38.15 0.003230
#> 51  2 38.16 0.002943
#> 52  2 38.17 0.002682
#> 53  2 38.18 0.002443
#> 54  2 38.18 0.002226
#> 55  2 38.19 0.002029
#> 56  2 38.19 0.001848
#> 57  2 38.20 0.001684
#> 58  2 38.20 0.001534
#> 59  2 38.20 0.001398
#> 60  2 38.21 0.001274
#> 61  2 38.21 0.001161
#> 62  2 38.21 0.001058
#> 63  2 38.21 0.000964
#> 64  2 38.21 0.000878
#> 65  2 38.21 0.000800
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(logistic_reg_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class1     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(logistic_reg_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.383       0.617 
#> 2        0.816       0.184 
#> 3        0.537       0.463 
#> 4        0.969       0.0313
#> 5        0.894       0.106 
#> 6        0.797       0.203
```
:::

## `h2o` 

This engine requires the agua extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(agua)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
logistic_reg_spec <- logistic_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("h2o")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
logistic_reg_fit <- logistic_reg_spec |>
  fit(class ~ ., data = bin_train)
logistic_reg_fit
#> parsnip model object
#> 
#> Model Details:
#> ==============
#> 
#> H2OBinomialModel: glm
#> Model ID:  GLM_model_R_1770287512312_5717 
#> GLM Model: summary
#>     family  link                                regularization
#> 1 binomial logit Elastic Net (alpha = 0.5, lambda = 6.162E-4 )
#>   number_of_predictors_total number_of_active_predictors number_of_iterations
#> 1                          2                           2                    4
#>      training_frame
#> 1 object_zkelygexok
#> 
#> Coefficients: glm coefficients
#>       names coefficients standardized_coefficients
#> 1 Intercept    -0.350788                 -0.350788
#> 2         A    -1.084233                 -1.084233
#> 3         B     2.759366                  2.759366
#> 
#> H2OBinomialMetrics: glm
#> ** Reported on training data. **
#> 
#> MSE:  0.130451
#> RMSE:  0.3611799
#> LogLoss:  0.4248206
#> Mean Per-Class Error:  0.1722728
#> AUC:  0.8889644
#> AUCPR:  0.8520865
#> Gini:  0.7779288
#> R^2:  0.4722968
#> Residual Deviance:  666.9684
#> AIC:  672.9684
#> 
#> Confusion Matrix (vertical: actual; across: predicted) for F1-optimal threshold:
#>        Class1 Class2    Error      Rate
#> Class1    350     84 0.193548   =84/434
#> Class2     53    298 0.150997   =53/351
#> Totals    403    382 0.174522  =137/785
#> 
#> Maximum Metrics: Maximum metrics at their respective thresholds
#>                         metric threshold      value idx
#> 1                       max f1  0.411045   0.813097 213
#> 2                       max f2  0.229916   0.868991 279
#> 3                 max f0point5  0.565922   0.816135 166
#> 4                 max accuracy  0.503565   0.826752 185
#> 5                max precision  0.997356   1.000000   0
#> 6                   max recall  0.009705   1.000000 395
#> 7              max specificity  0.997356   1.000000   0
#> 8             max absolute_mcc  0.411045   0.652014 213
#> 9   max min_per_class_accuracy  0.454298   0.822581 201
#> 10 max mean_per_class_accuracy  0.411045   0.827727 213
#> 11                     max tns  0.997356 434.000000   0
#> 12                     max fns  0.997356 349.000000   0
#> 13                     max fps  0.001723 434.000000 399
#> 14                     max tps  0.009705 351.000000 395
#> 15                     max tnr  0.997356   1.000000   0
#> 16                     max fnr  0.997356   0.994302   0
#> 17                     max fpr  0.001723   1.000000 399
#> 18                     max tpr  0.009705   1.000000 395
#> 
#> Gains/Lift Table: Extract with `h2o.gainsLift(<model>, <data>)` or `h2o.gainsLift(<model>, valid=<T/F>, xval=<T/F>)`
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(logistic_reg_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(logistic_reg_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.399       0.601 
#> 2        0.857       0.143 
#> 3        0.540       0.460 
#> 4        0.976       0.0243
#> 5        0.908       0.0925
#> 6        0.848       0.152
```
:::

## `keras` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
logistic_reg_spec <- logistic_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("keras")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(730)
logistic_reg_fit <- logistic_reg_spec |>
  fit(class ~ ., data = bin_train)
```
:::

::: {.cell layout-align="center"}

```{.r .cell-code}
logistic_reg_fit
#> parsnip model object
#> 
#> Model: "sequential"
#> ________________________________________________________________________________
#>  Layer (type)                       Output Shape                    Param #     
#> ================================================================================
#>  dense (Dense)                      (None, 1)                       3           
#>  dense_1 (Dense)                    (None, 2)                       4           
#> ================================================================================
#> Total params: 7 (28.00 Byte)
#> Trainable params: 7 (28.00 Byte)
#> Non-trainable params: 0 (0.00 Byte)
#> ________________________________________________________________________________
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(logistic_reg_fit, type = "class", new_data = bin_test)
#> 1/1 - 0s - 81ms/epoch - 81ms/step
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class1     
#> 4 Class1     
#> 5 Class1     
#> 6 Class2
predict(logistic_reg_fit, type = "prob", new_data = bin_test)
#> 1/1 - 0s - 6ms/epoch - 6ms/step
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.212       0.788 
#> 2        0.627       0.373 
#> 3        0.580       0.420 
#> 4        0.990       0.0101
#> 5        0.954       0.0461
#> 6        0.470       0.530
```
:::

## `LiblineaR` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
logistic_reg_spec <- logistic_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("LiblineaR")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
logistic_reg_fit <- logistic_reg_spec |>
  fit(class ~ ., data = bin_train)
logistic_reg_fit
#> parsnip model object
#> 
#> $TypeDetail
#> [1] "L2-regularized logistic regression primal (L2R_LR)"
#> 
#> $Type
#> [1] 0
#> 
#> $W
#>             A        B      Bias
#> [1,] 1.014233 -2.65166 0.3363362
#> 
#> $Bias
#> [1] 1
#> 
#> $ClassNames
#> [1] Class1 Class2
#> Levels: Class1 Class2
#> 
#> $NbClass
#> [1] 2
#> 
#> attr(,"class")
#> [1] "LiblineaR"
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(logistic_reg_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class1     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(logistic_reg_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.397       0.603 
#> 2        0.847       0.153 
#> 3        0.539       0.461 
#> 4        0.973       0.0267
#> 5        0.903       0.0974
#> 6        0.837       0.163
```
:::

## `stan` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
logistic_reg_spec <- logistic_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("stan")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(96)
logistic_reg_fit <- 
  logistic_reg_spec |> 
  fit(outcome ~ treatment * visit, data = cls_group_train)
logistic_reg_fit |> print(digits = 3)
#> parsnip model object
#> 
#> stan_glm
#>  family:       binomial [logit]
#>  formula:      outcome ~ treatment * visit
#>  observations: 1433
#>  predictors:   4
#> ------
#>                            Median MAD_SD
#> (Intercept)                -0.137  0.187
#> treatmentterbinafine       -0.108  0.264
#> visit                      -0.335  0.050
#> treatmentterbinafine:visit -0.048  0.073
#> 
#> ------
#> * For help interpreting the printed output see ?print.stanreg
#> * For info on the priors used see ?prior_summary.stanreg
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(logistic_reg_fit, type = "class", new_data = cls_group_test)
#> # A tibble: 475 × 1
#>    .pred_class 
#>    <fct>       
#>  1 none or mild
#>  2 none or mild
#>  3 none or mild
#>  4 none or mild
#>  5 none or mild
#>  6 none or mild
#>  7 none or mild
#>  8 none or mild
#>  9 none or mild
#> 10 none or mild
#> # ℹ 465 more rows
predict(logistic_reg_fit, type = "prob", new_data = cls_group_test)
#> # A tibble: 475 × 2
#>    `.pred_none or mild` `.pred_moderate or severe`
#>                   <dbl>                      <dbl>
#>  1                0.652                     0.348 
#>  2                0.734                     0.266 
#>  3                0.802                     0.198 
#>  4                0.856                     0.144 
#>  5                0.898                     0.102 
#>  6                0.928                     0.0721
#>  7                0.950                     0.0502
#>  8                0.617                     0.383 
#>  9                0.692                     0.308 
#> 10                0.759                     0.241 
#> # ℹ 465 more rows
predict(logistic_reg_fit, type = "conf_int", new_data = cls_group_test)
#> # A tibble: 475 × 4
#>    `.pred_lower_none or mild` `.pred_upper_none or mild` .pred_lower_moderate …¹
#>                         <dbl>                      <dbl>                   <dbl>
#>  1                      0.583                      0.715                  0.285 
#>  2                      0.689                      0.776                  0.224 
#>  3                      0.771                      0.832                  0.168 
#>  4                      0.827                      0.883                  0.117 
#>  5                      0.868                      0.924                  0.0761
#>  6                      0.899                      0.952                  0.0482
#>  7                      0.922                      0.970                  0.0302
#>  8                      0.547                      0.683                  0.317 
#>  9                      0.644                      0.736                  0.264 
#> 10                      0.723                      0.791                  0.209 
#> # ℹ 465 more rows
#> # ℹ abbreviated name: ¹​`.pred_lower_moderate or severe`
#> # ℹ 1 more variable: `.pred_upper_moderate or severe` <dbl>
predict(logistic_reg_fit, type = "pred_int", new_data = cls_group_test)
#> # A tibble: 475 × 4
#>    `.pred_lower_none or mild` `.pred_upper_none or mild` .pred_lower_moderate …¹
#>                         <dbl>                      <dbl>                   <dbl>
#>  1                          0                          1                       0
#>  2                          0                          1                       0
#>  3                          0                          1                       0
#>  4                          0                          1                       0
#>  5                          0                          1                       0
#>  6                          0                          1                       0
#>  7                          0                          1                       0
#>  8                          0                          1                       0
#>  9                          0                          1                       0
#> 10                          0                          1                       0
#> # ℹ 465 more rows
#> # ℹ abbreviated name: ¹​`.pred_lower_moderate or severe`
#> # ℹ 1 more variable: `.pred_upper_moderate or severe` <dbl>
```
:::

## `stan_glmer` 

This engine requires the multilevelmod extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(multilevelmod)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
logistic_reg_spec <- logistic_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("stan_glmer")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(484)
logistic_reg_fit <- 
  logistic_reg_spec |> 
  fit(outcome ~ treatment * visit + (1 | patientID), data = cls_group_train)
logistic_reg_fit |> print(digits = 3)
#> parsnip model object
#> 
#> stan_glmer
#>  family:       binomial [logit]
#>  formula:      outcome ~ treatment * visit + (1 | patientID)
#>  observations: 1433
#> ------
#>                            Median MAD_SD
#> (Intercept)                -0.628  0.585
#> treatmentterbinafine       -0.686  0.821
#> visit                      -0.830  0.105
#> treatmentterbinafine:visit -0.023  0.143
#> 
#> Error terms:
#>  Groups    Name        Std.Dev.
#>  patientID (Intercept) 4.376   
#> Num. levels: patientID 219 
#> 
#> ------
#> * For help interpreting the printed output see ?print.stanreg
#> * For info on the priors used see ?prior_summary.stanreg
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(logistic_reg_fit, type = "class", new_data = cls_group_test)
#> # A tibble: 475 × 1
#>    .pred_class 
#>    <fct>       
#>  1 none or mild
#>  2 none or mild
#>  3 none or mild
#>  4 none or mild
#>  5 none or mild
#>  6 none or mild
#>  7 none or mild
#>  8 none or mild
#>  9 none or mild
#> 10 none or mild
#> # ℹ 465 more rows
predict(logistic_reg_fit, type = "prob", new_data = cls_group_test)
#> # A tibble: 475 × 2
#>    `.pred_none or mild` `.pred_moderate or severe`
#>                   <dbl>                      <dbl>
#>  1                0.671                     0.329 
#>  2                0.730                     0.270 
#>  3                0.796                     0.204 
#>  4                0.847                     0.153 
#>  5                0.882                     0.118 
#>  6                0.909                     0.0908
#>  7                0.934                     0.0655
#>  8                0.613                     0.387 
#>  9                0.681                     0.319 
#> 10                0.744                     0.256 
#> # ℹ 465 more rows
predict(logistic_reg_fit, type = "conf_int", new_data = cls_group_test)
#> # A tibble: 475 × 4
#>    `.pred_lower_none or mild` `.pred_upper_none or mild` .pred_lower_moderate …¹
#>                         <dbl>                      <dbl>                   <dbl>
#>  1                   0.00184                       1.000             0.0000217  
#>  2                   0.00417                       1.000             0.00000942 
#>  3                   0.00971                       1.000             0.00000412 
#>  4                   0.0214                        1.000             0.00000169 
#>  5                   0.0465                        1.000             0.000000706
#>  6                   0.101                         1.000             0.000000300
#>  7                   0.203                         1.000             0.000000120
#>  8                   0.000923                      1.000             0.0000440  
#>  9                   0.00196                       1.000             0.0000175  
#> 10                   0.00447                       1.000             0.00000724 
#> # ℹ 465 more rows
#> # ℹ abbreviated name: ¹​`.pred_lower_moderate or severe`
#> # ℹ 1 more variable: `.pred_upper_moderate or severe` <dbl>
predict(logistic_reg_fit, type = "pred_int", new_data = cls_group_test)
#> # A tibble: 475 × 4
#>    `.pred_lower_none or mild` `.pred_upper_none or mild` .pred_lower_moderate …¹
#>                         <dbl>                      <dbl>                   <dbl>
#>  1                          0                          1                       0
#>  2                          0                          1                       0
#>  3                          0                          1                       0
#>  4                          0                          1                       0
#>  5                          0                          1                       0
#>  6                          0                          1                       0
#>  7                          0                          1                       0
#>  8                          0                          1                       0
#>  9                          0                          1                       0
#> 10                          0                          1                       0
#> # ℹ 465 more rows
#> # ℹ abbreviated name: ¹​`.pred_lower_moderate or severe`
#> # ℹ 1 more variable: `.pred_upper_moderate or severe` <dbl>
```
:::

## `spark` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
logistic_reg_spec <- logistic_reg() |> 
  set_engine("spark")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
logistic_reg_fit <- logistic_reg_spec |>
  fit(Class ~ ., data = tbl_bin$training)
logistic_reg_fit
#> parsnip model object
#> 
#> Formula: Class ~ .
#> 
#> Coefficients:
#> (Intercept)           A           B 
#>   -3.731170   -1.214355    3.794186
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(logistic_reg_fit, type = "class", new_data = tbl_bin$test)
#> # Source:   SQL [?? x 1]
#> # Database: spark_connection
#>   pred_class
#>   <chr>     
#> 1 Class2    
#> 2 Class2    
#> 3 Class1    
#> 4 Class2    
#> 5 Class2    
#> 6 Class1    
#> 7 Class2
predict(logistic_reg_fit, type = "prob", new_data = tbl_bin$test)
#> # Source:   SQL [?? x 2]
#> # Database: spark_connection
#>   pred_Class1 pred_Class2
#>         <dbl>       <dbl>
#> 1       0.130       0.870
#> 2       0.262       0.738
#> 3       0.787       0.213
#> 4       0.279       0.721
#> 5       0.498       0.502
#> 6       0.900       0.100
#> 7       0.161       0.839
```
:::

:::

### Multivariate Adaptive Regression Splines (`mars()`)

:::{.panel-tabset}

## `earth`

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
mars_spec <- mars() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because earth is the default.
  set_mode("classification")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
mars_fit <- mars_spec |>
  fit(class ~ ., data = bin_train)
mars_fit
#> parsnip model object
#> 
#> GLM (family binomial, link logit):
#>  nulldev  df       dev  df   devratio     AIC iters converged
#>  1079.45 784   638.975 779      0.408     651     5         1
#> 
#> Earth selected 6 of 13 terms, and 2 of 2 predictors
#> Termination condition: Reached nk 21
#> Importance: B, A
#> Number of terms at each degree of interaction: 1 5 (additive model)
#> Earth GCV 0.1342746    RSS 102.4723    GRSq 0.4582121    RSq 0.4719451
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(mars_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(mars_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.410       0.590 
#> 2        0.794       0.206 
#> 3        0.356       0.644 
#> 4        0.927       0.0729
#> 5        0.927       0.0729
#> 6        0.836       0.164
```
:::

:::

### Neural Networks (`mlp()`)

:::{.panel-tabset}

## `nnet`

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
mlp_spec <- mlp() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because nnet is the default.
  set_mode("classification")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(839)
mlp_fit <- mlp_spec |>
  fit(class ~ ., data = bin_train)
mlp_fit
#> parsnip model object
#> 
#> a 2-5-1 network with 21 weights
#> inputs: A B 
#> output(s): class 
#> options were - entropy fitting
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(mlp_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(mlp_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.390        0.610
#> 2        0.685        0.315
#> 3        0.433        0.567
#> 4        0.722        0.278
#> 5        0.720        0.280
#> 6        0.684        0.316
```
:::

## `brulee` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
mlp_spec <- mlp() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("classification") |>
  set_engine("brulee")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(38)
mlp_fit <- mlp_spec |>
  fit(class ~ ., data = bin_train)
mlp_fit
#> parsnip model object
#> 
#> Multilayer perceptron
#> 
#> relu activation,
#> 3 hidden units,
#> 17 model parameters
#> 785 samples, 2 features, 2 classes 
#> class weights Class1=1, Class2=1 
#> weight decay: 0.001 
#> dropout proportion: 0 
#> batch size: 707 
#> learn rate: 0.01 
#> validation loss after 5 epochs: 0.427
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(mlp_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class1     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(mlp_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.387       0.613 
#> 2        0.854       0.146 
#> 3        0.540       0.460 
#> 4        0.941       0.0589
#> 5        0.882       0.118 
#> 6        0.842       0.158
```
:::

## `brulee_two_layer` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
mlp_spec <- mlp() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("classification") |>
  set_engine("brulee_two_layer")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(336)
mlp_fit <- mlp_spec |>
  fit(class ~ ., data = bin_train)
mlp_fit
#> parsnip model object
#> 
#> Multilayer perceptron
#> 
#> c(relu,relu) activation,
#> c(3,3) hidden units,
#> 29 model parameters
#> 785 samples, 2 features, 2 classes 
#> class weights Class1=1, Class2=1 
#> weight decay: 0.001 
#> dropout proportion: 0 
#> batch size: 707 
#> learn rate: 0.01 
#> validation loss after 17 epochs: 0.405
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(mlp_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(mlp_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.392       0.608 
#> 2        0.835       0.165 
#> 3        0.440       0.560 
#> 4        0.938       0.0620
#> 5        0.938       0.0620
#> 6        0.848       0.152
```
:::

## `h2o` 

This engine requires the agua extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(agua)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
mlp_spec <- mlp() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("classification") |>
  set_engine("h2o")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(306)
mlp_fit <- mlp_spec |>
  fit(class ~ ., data = bin_train)
mlp_fit
#> parsnip model object
#> 
#> Model Details:
#> ==============
#> 
#> H2OBinomialModel: deeplearning
#> Model ID:  DeepLearning_model_R_1770287512312_5719 
#> Status of Neuron Layers: predicting .outcome, 2-class classification, bernoulli distribution, CrossEntropy loss, 1,002 weights/biases, 16.9 KB, 7,850 training samples, mini-batch size 1
#>   layer units      type dropout       l1       l2 mean_rate rate_rms momentum
#> 1     1     2     Input  0.00 %       NA       NA        NA       NA       NA
#> 2     2   200 Rectifier  0.00 % 0.000000 0.000000  0.008355 0.020870 0.000000
#> 3     3     2   Softmax      NA 0.000000 0.000000  0.003311 0.000205 0.000000
#>   mean_weight weight_rms mean_bias bias_rms
#> 1          NA         NA        NA       NA
#> 2    0.001630   0.103344  0.489596 0.030962
#> 3   -0.001547   0.402480 -0.040452 0.052851
#> 
#> 
#> H2OBinomialMetrics: deeplearning
#> ** Reported on training data. **
#> ** Metrics reported on full training frame **
#> 
#> MSE:  0.2977407
#> RMSE:  0.5456562
#> LogLoss:  0.9645075
#> Mean Per-Class Error:  0.1762509
#> AUC:  0.8895913
#> AUCPR:  0.8505352
#> Gini:  0.7791826
#> 
#> Confusion Matrix (vertical: actual; across: predicted) for F1-optimal threshold:
#>        Class1 Class2    Error      Rate
#> Class1    328    106 0.244240  =106/434
#> Class2     38    313 0.108262   =38/351
#> Totals    366    419 0.183439  =144/785
#> 
#> Maximum Metrics: Maximum metrics at their respective thresholds
#>                         metric threshold      value idx
#> 1                       max f1  0.027001   0.812987 311
#> 2                       max f2  0.014770   0.869336 342
#> 3                 max f0point5  0.076925   0.816191 240
#> 4                 max accuracy  0.052351   0.828025 273
#> 5                max precision  0.911921   1.000000   0
#> 6                   max recall  0.000422   1.000000 397
#> 7              max specificity  0.911921   1.000000   0
#> 8             max absolute_mcc  0.051136   0.651742 274
#> 9   max min_per_class_accuracy  0.044334   0.820513 282
#> 10 max mean_per_class_accuracy  0.051136   0.825400 274
#> 11                     max tns  0.911921 434.000000   0
#> 12                     max fns  0.911921 350.000000   0
#> 13                     max fps  0.000061 434.000000 399
#> 14                     max tps  0.000422 351.000000 397
#> 15                     max tnr  0.911921   1.000000   0
#> 16                     max fnr  0.911921   0.997151   0
#> 17                     max fpr  0.000061   1.000000 399
#> 18                     max tpr  0.000422   1.000000 397
#> 
#> Gains/Lift Table: Extract with `h2o.gainsLift(<model>, <data>)` or `h2o.gainsLift(<model>, valid=<T/F>, xval=<T/F>)`
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(mlp_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(mlp_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.930      0.0702 
#> 2        0.992      0.00786
#> 3        0.957      0.0429 
#> 4        0.999      0.00128
#> 5        0.995      0.00534
#> 6        0.992      0.00821
```
:::

## `keras` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
mlp_spec <- mlp() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("classification") |>
  set_engine("keras")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(216)
mlp_fit <- mlp_spec |>
  fit(class ~ ., data = bin_train)
```
:::

::: {.cell layout-align="center"}

```{.r .cell-code}
mlp_fit
#> parsnip model object
#> 
#> Model: "sequential_1"
#> ________________________________________________________________________________
#>  Layer (type)                       Output Shape                    Param #     
#> ================================================================================
#>  dense_2 (Dense)                    (None, 5)                       15          
#>  dense_3 (Dense)                    (None, 2)                       12          
#> ================================================================================
#> Total params: 27 (108.00 Byte)
#> Trainable params: 27 (108.00 Byte)
#> Non-trainable params: 0 (0.00 Byte)
#> ________________________________________________________________________________
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(mlp_fit, type = "class", new_data = bin_test)
#> 1/1 - 0s - 38ms/epoch - 38ms/step
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class1     
#> 4 Class1     
#> 5 Class1     
#> 6 Class2
predict(mlp_fit, type = "prob", new_data = bin_test)
#> 1/1 - 0s - 6ms/epoch - 6ms/step
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.315        0.685
#> 2        0.584        0.416
#> 3        0.508        0.492
#> 4        0.896        0.104
#> 5        0.871        0.129
#> 6        0.475        0.525
```
:::

:::

### Multinom Regression (`multinom_reg()`)

:::{.panel-tabset}

## `nnet` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
# This model only works with a single mode, so we don't need to set the mode.
# We don't need to set the engine because nnet is the default.
multinom_reg_spec <- multinom_reg()
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(634)
multinom_reg_fit <- multinom_reg_spec |>
  fit(class ~ ., data = mtl_train)
multinom_reg_fit
#> parsnip model object
#> 
#> Call:
#> nnet::multinom(formula = class ~ ., data = data, trace = FALSE)
#> 
#> Coefficients:
#>       (Intercept)        A        B
#> two    -0.5868435 1.881920 1.379106
#> three   0.2910810 1.129622 1.292802
#> 
#> Residual Deviance: 315.8164 
#> AIC: 327.8164
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(multinom_reg_fit, type = "class", new_data = mtl_test)
#> # A tibble: 8 × 1
#>   .pred_class
#>   <fct>      
#> 1 three      
#> 2 three      
#> 3 three      
#> 4 one        
#> 5 one        
#> 6 two        
#> 7 three      
#> 8 one
predict(multinom_reg_fit, type = "prob", new_data = mtl_test)
#> # A tibble: 8 × 3
#>   .pred_one .pred_two .pred_three
#>       <dbl>     <dbl>       <dbl>
#> 1   0.145     0.213        0.641 
#> 2   0.308     0.178        0.514 
#> 3   0.350     0.189        0.461 
#> 4   0.983     0.00123      0.0155
#> 5   0.956     0.00275      0.0415
#> 6   0.00318   0.754        0.243 
#> 7   0.0591    0.414        0.527 
#> 8   0.522     0.0465       0.431
```
:::

## `brulee` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
multinom_reg_spec <- multinom_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("brulee")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(837)
multinom_reg_fit <- multinom_reg_spec |>
  fit(class ~ ., data = mtl_train)
multinom_reg_fit
#> parsnip model object
#> 
#> Multinomial regression
#> 
#> 192 samples, 2 features, 3 classes 
#> class weights one=1, two=1, three=1 
#> weight decay: 0.001 
#> batch size: 173 
#> validation loss after 1 epoch: 0.953
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(multinom_reg_fit, type = "class", new_data = mtl_test)
#> # A tibble: 8 × 1
#>   .pred_class
#>   <fct>      
#> 1 three      
#> 2 three      
#> 3 three      
#> 4 one        
#> 5 one        
#> 6 two        
#> 7 three      
#> 8 three
predict(multinom_reg_fit, type = "prob", new_data = mtl_test)
#> # A tibble: 8 × 3
#>   .pred_one .pred_two .pred_three
#>       <dbl>     <dbl>       <dbl>
#> 1   0.131     0.190        0.679 
#> 2   0.303     0.174        0.523 
#> 3   0.358     0.192        0.449 
#> 4   0.983     0.00125      0.0154
#> 5   0.948     0.00275      0.0491
#> 6   0.00344   0.796        0.200 
#> 7   0.0611    0.420        0.518 
#> 8   0.443     0.0390       0.518
```
:::

## `glmnet` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
multinom_reg_spec <- multinom_reg(penalty = 0.01) |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("glmnet")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
multinom_reg_fit <- multinom_reg_spec |>
  fit(class ~ ., data = mtl_train)
multinom_reg_fit
#> parsnip model object
#> 
#> 
#> Call:  glmnet::glmnet(x = maybe_matrix(x), y = y, family = "multinomial") 
#> 
#>    Df  %Dev   Lambda
#> 1   0  0.00 0.219200
#> 2   1  1.61 0.199700
#> 3   2  3.90 0.181900
#> 4   2  6.07 0.165800
#> 5   2  7.93 0.151100
#> 6   2  9.52 0.137600
#> 7   2 10.90 0.125400
#> 8   2 12.09 0.114300
#> 9   2 13.13 0.104100
#> 10  2 14.22 0.094870
#> 11  2 15.28 0.086440
#> 12  2 16.20 0.078760
#> 13  2 16.99 0.071760
#> 14  2 17.68 0.065390
#> 15  2 18.28 0.059580
#> 16  2 18.80 0.054290
#> 17  2 19.24 0.049460
#> 18  2 19.63 0.045070
#> 19  2 19.96 0.041070
#> 20  2 20.25 0.037420
#> 21  2 20.49 0.034090
#> 22  2 20.70 0.031070
#> 23  2 20.88 0.028310
#> 24  2 21.04 0.025790
#> 25  2 21.17 0.023500
#> 26  2 21.28 0.021410
#> 27  2 21.38 0.019510
#> 28  2 21.46 0.017780
#> 29  2 21.53 0.016200
#> 30  2 21.58 0.014760
#> 31  2 21.63 0.013450
#> 32  2 21.67 0.012250
#> 33  2 21.71 0.011160
#> 34  2 21.74 0.010170
#> 35  2 21.77 0.009269
#> 36  2 21.79 0.008445
#> 37  2 21.82 0.007695
#> 38  2 21.83 0.007011
#> 39  2 21.85 0.006389
#> 40  2 21.86 0.005821
#> 41  2 21.87 0.005304
#> 42  2 21.88 0.004833
#> 43  2 21.89 0.004403
#> 44  2 21.89 0.004012
#> 45  2 21.90 0.003656
#> 46  2 21.90 0.003331
#> 47  2 21.91 0.003035
#> 48  2 21.91 0.002765
#> 49  2 21.91 0.002520
#> 50  2 21.91 0.002296
#> 51  2 21.92 0.002092
#> 52  2 21.92 0.001906
#> 53  2 21.92 0.001737
#> 54  2 21.92 0.001582
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(multinom_reg_fit, type = "class", new_data = mtl_test)
#> # A tibble: 8 × 1
#>   .pred_class
#>   <fct>      
#> 1 three      
#> 2 three      
#> 3 three      
#> 4 one        
#> 5 one        
#> 6 two        
#> 7 three      
#> 8 one
predict(multinom_reg_fit, type = "prob", new_data = mtl_test)
#> # A tibble: 8 × 3
#>   .pred_one .pred_two .pred_three
#>       <dbl>     <dbl>       <dbl>
#> 1   0.163     0.211        0.626 
#> 2   0.318     0.185        0.496 
#> 3   0.358     0.198        0.444 
#> 4   0.976     0.00268      0.0217
#> 5   0.940     0.00529      0.0544
#> 6   0.00617   0.699        0.295 
#> 7   0.0757    0.390        0.534 
#> 8   0.506     0.0563       0.438
```
:::

## `h2o` 

This engine requires the agua extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(agua)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
multinom_reg_spec <- multinom_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("h2o")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
multinom_reg_fit <- multinom_reg_spec |>
  fit(class ~ ., data = mtl_train)
multinom_reg_fit
#> parsnip model object
#> 
#> Model Details:
#> ==============
#> 
#> H2OMultinomialModel: glm
#> Model ID:  GLM_model_R_1770287512312_5722 
#> GLM Model: summary
#>        family        link                                regularization
#> 1 multinomial multinomial Elastic Net (alpha = 0.5, lambda = 4.372E-4 )
#>   number_of_predictors_total number_of_active_predictors number_of_iterations
#> 1                          9                           6                    4
#>      training_frame
#> 1 object_jbhwnlsrno
#> 
#> Coefficients: glm multinomial coefficients
#>       names coefs_class_0 coefs_class_1 coefs_class_2 std_coefs_class_0
#> 1 Intercept     -1.119482     -0.831434     -1.706488         -1.083442
#> 2         A     -1.119327      0.002894      0.750746         -1.029113
#> 3         B     -1.208210      0.078752      0.162842         -1.187423
#>   std_coefs_class_1 std_coefs_class_2
#> 1         -0.819868         -1.830487
#> 2          0.002661          0.690238
#> 3          0.077397          0.160041
#> 
#> H2OMultinomialMetrics: glm
#> ** Reported on training data. **
#> 
#> Training Set Metrics: 
#> =====================
#> 
#> Extract training frame with `h2o.getFrame("object_jbhwnlsrno")`
#> MSE: (Extract with `h2o.mse`) 0.2982118
#> RMSE: (Extract with `h2o.rmse`) 0.5460878
#> Logloss: (Extract with `h2o.logloss`) 0.822443
#> Mean Per-Class Error: 0.4583896
#> AUC: (Extract with `h2o.auc`) NaN
#> AUCPR: (Extract with `h2o.aucpr`) NaN
#> Null Deviance: (Extract with `h2o.nulldeviance`) 404.5036
#> Residual Deviance: (Extract with `h2o.residual_deviance`) 315.8181
#> R^2: (Extract with `h2o.r2`) 0.4682043
#> AIC: (Extract with `h2o.aic`) NaN
#> Confusion Matrix: Extract with `h2o.confusionMatrix(<model>,train = TRUE)`)
#> =========================================================================
#> Confusion Matrix: Row labels: Actual class; Column labels: Predicted class
#>        one three two  Error       Rate
#> one     59    18   1 0.2436 =  19 / 78
#> three   19    52   5 0.3158 =  24 / 76
#> two      7    24   7 0.8158 =  31 / 38
#> Totals  85    94  13 0.3854 = 74 / 192
#> 
#> Hit Ratio Table: Extract with `h2o.hit_ratio_table(<model>,train = TRUE)`
#> =======================================================================
#> Top-3 Hit Ratios: 
#>   k hit_ratio
#> 1 1  0.614583
#> 2 2  0.890625
#> 3 3  1.000000
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(multinom_reg_fit, type = "class", new_data = mtl_test)
#> # A tibble: 8 × 1
#>   .pred_class
#>   <fct>      
#> 1 three      
#> 2 three      
#> 3 three      
#> 4 one        
#> 5 one        
#> 6 two        
#> 7 three      
#> 8 one
predict(multinom_reg_fit, type = "prob", new_data = mtl_test)
#> # A tibble: 8 × 3
#>   .pred_one .pred_three .pred_two
#>       <dbl>       <dbl>     <dbl>
#> 1   0.146        0.641    0.213  
#> 2   0.308        0.513    0.179  
#> 3   0.350        0.460    0.190  
#> 4   0.983        0.0158   0.00128
#> 5   0.955        0.0422   0.00284
#> 6   0.00329      0.244    0.752  
#> 7   0.0599       0.527    0.413  
#> 8   0.521        0.432    0.0469
```
:::

## `keras` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
multinom_reg_spec <- multinom_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("keras")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
multinom_reg_fit <- multinom_reg_spec |>
  fit(class ~ ., data = mtl_train)
```
:::

::: {.cell layout-align="center"}

```{.r .cell-code}
multinom_reg_fit
#> parsnip model object
#> 
#> Model: "sequential_2"
#> ________________________________________________________________________________
#>  Layer (type)                       Output Shape                    Param #     
#> ================================================================================
#>  dense_4 (Dense)                    (None, 1)                       3           
#>  dense_5 (Dense)                    (None, 3)                       6           
#> ================================================================================
#> Total params: 9 (36.00 Byte)
#> Trainable params: 9 (36.00 Byte)
#> Non-trainable params: 0 (0.00 Byte)
#> ________________________________________________________________________________
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(multinom_reg_fit, type = "class", new_data = mtl_test)
#> 1/1 - 0s - 38ms/epoch - 38ms/step
#> # A tibble: 8 × 1
#>   .pred_class
#>   <fct>      
#> 1 three      
#> 2 three      
#> 3 one        
#> 4 one        
#> 5 one        
#> 6 three      
#> 7 three      
#> 8 one
predict(multinom_reg_fit, type = "prob", new_data = mtl_test)
#> 1/1 - 0s - 5ms/epoch - 5ms/step
#> # A tibble: 8 × 3
#>   .pred_one .pred_two .pred_three
#>       <dbl>     <dbl>       <dbl>
#> 1    0.261      0.342      0.397 
#> 2    0.335      0.326      0.339 
#> 3    0.352      0.322      0.326 
#> 4    0.753      0.156      0.0914
#> 5    0.683      0.192      0.126 
#> 6    0.0913     0.336      0.572 
#> 7    0.202      0.349      0.449 
#> 8    0.418      0.302      0.281
```
:::

## `spark` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
multinom_reg_spec <- multinom_reg() |> 
  set_engine("spark")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
multinom_reg_fit <- multinom_reg_spec |>
  fit(class ~ ., data = tbl_mtl$training)
multinom_reg_fit
#> parsnip model object
#> 
#> Formula: class ~ .
#> 
#> Coefficients:
#>       (Intercept)          A          B
#> one    0.05447853 -1.0569131 -0.9049194
#> three  0.41207949  0.1458870  0.3959664
#> two   -0.46655802  0.9110261  0.5089529
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(multinom_reg_fit, type = "class", new_data = tbl_mtl$test)
#> # Source:   SQL [?? x 1]
#> # Database: spark_connection
#>   pred_class
#>   <chr>     
#> 1 one       
#> 2 one       
#> 3 three     
#> 4 three     
#> 5 three     
#> 6 three     
#> 7 three
predict(multinom_reg_fit, type = "prob", new_data = tbl_mtl$test)
#> # Source:   SQL [?? x 3]
#> # Database: spark_connection
#>   pred_one pred_three pred_two
#>      <dbl>      <dbl>    <dbl>
#> 1   0.910      0.0814  0.00904
#> 2   0.724      0.233   0.0427 
#> 3   0.124      0.620   0.256  
#> 4   0.0682     0.610   0.322  
#> 5   0.130      0.571   0.300  
#> 6   0.115      0.549   0.336  
#> 7   0.0517     0.524   0.424
```
:::

:::

### Naive Bayes (`naive_Bayes()`)

:::{.panel-tabset}

## `h2o` 

This engine requires the agua extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(agua)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
naive_Bayes_spec <- naive_Bayes() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("h2o")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
naive_Bayes_fit <- naive_Bayes_spec |>
  fit(class ~ ., data = bin_train)
naive_Bayes_fit
#> parsnip model object
#> 
#> Model Details:
#> ==============
#> 
#> H2OBinomialModel: naivebayes
#> Model ID:  NaiveBayes_model_R_1770287512312_5723 
#> Model Summary: 
#>   number_of_response_levels min_apriori_probability max_apriori_probability
#> 1                         2                 0.44713                 0.55287
#> 
#> 
#> H2OBinomialMetrics: naivebayes
#> ** Reported on training data. **
#> 
#> MSE:  0.1737113
#> RMSE:  0.4167869
#> LogLoss:  0.5473431
#> Mean Per-Class Error:  0.2356138
#> AUC:  0.8377152
#> AUCPR:  0.788608
#> Gini:  0.6754303
#> 
#> Confusion Matrix (vertical: actual; across: predicted) for F1-optimal threshold:
#>        Class1 Class2    Error      Rate
#> Class1    274    160 0.368664  =160/434
#> Class2     36    315 0.102564   =36/351
#> Totals    310    475 0.249682  =196/785
#> 
#> Maximum Metrics: Maximum metrics at their respective thresholds
#>                         metric threshold      value idx
#> 1                       max f1  0.175296   0.762712 286
#> 2                       max f2  0.133412   0.851119 306
#> 3                 max f0point5  0.497657   0.731343 183
#> 4                 max accuracy  0.281344   0.765605 248
#> 5                max precision  0.999709   1.000000   0
#> 6                   max recall  0.020983   1.000000 390
#> 7              max specificity  0.999709   1.000000   0
#> 8             max absolute_mcc  0.280325   0.541898 249
#> 9   max min_per_class_accuracy  0.398369   0.758065 215
#> 10 max mean_per_class_accuracy  0.280325   0.771945 249
#> 11                     max tns  0.999709 434.000000   0
#> 12                     max fns  0.999709 347.000000   0
#> 13                     max fps  0.006522 434.000000 399
#> 14                     max tps  0.020983 351.000000 390
#> 15                     max tnr  0.999709   1.000000   0
#> 16                     max fnr  0.999709   0.988604   0
#> 17                     max fpr  0.006522   1.000000 399
#> 18                     max tpr  0.020983   1.000000 390
#> 
#> Gains/Lift Table: Extract with `h2o.gainsLift(<model>, <data>)` or `h2o.gainsLift(<model>, valid=<T/F>, xval=<T/F>)`
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(naive_Bayes_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class2     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class2
predict(naive_Bayes_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.181      0.819  
#> 2        0.750      0.250  
#> 3        0.556      0.444  
#> 4        0.994      0.00643
#> 5        0.967      0.0331 
#> 6        0.630      0.370
```
:::

## `klaR` 

This engine requires the discrim extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(discrim)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
# This model only works with a single mode, so we don't need to set the mode.
# We don't need to set the engine because klaR is the default.
naive_Bayes_spec <- naive_Bayes()
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
naive_Bayes_fit <- naive_Bayes_spec |>
  fit(class ~ ., data = bin_train)
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(naive_Bayes_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(naive_Bayes_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.250      0.750  
#> 2        0.593      0.407  
#> 3        0.333      0.667  
#> 4        0.993      0.00658
#> 5        0.978      0.0223 
#> 6        0.531      0.469
```
:::

## `naivebayes` 

This engine requires the discrim extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(discrim)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
naive_Bayes_spec <- naive_Bayes() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("naivebayes")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
naive_Bayes_fit <- naive_Bayes_spec |>
  fit(class ~ ., data = bin_train)
naive_Bayes_fit
#> parsnip model object
#> 
#> 
#> ================================= Naive Bayes ==================================
#> 
#> Call:
#> naive_bayes.default(x = maybe_data_frame(x), y = y, usekernel = TRUE)
#> 
#> -------------------------------------------------------------------------------- 
#>  
#> Laplace smoothing: 0
#> 
#> -------------------------------------------------------------------------------- 
#>  
#> A priori probabilities: 
#> 
#>    Class1    Class2 
#> 0.5528662 0.4471338 
#> 
#> -------------------------------------------------------------------------------- 
#>  
#> Tables: 
#> 
#> -------------------------------------------------------------------------------- 
#> :: A::Class1 (KDE)
#> -------------------------------------------------------------------------------- 
#> 
#> Call:
#> 	density.default(x = x, na.rm = TRUE)
#> 
#> Data: x (434 obs.);	Bandwidth 'bw' = 0.2548
#> 
#>        x                 y            
#>  Min.   :-2.5638   Min.   :0.0002915  
#>  1st Qu.:-1.2013   1st Qu.:0.0506201  
#>  Median : 0.1612   Median :0.1619843  
#>  Mean   : 0.1612   Mean   :0.1831190  
#>  3rd Qu.: 1.5237   3rd Qu.:0.2581668  
#>  Max.   : 2.8862   Max.   :0.5370762  
#> -------------------------------------------------------------------------------- 
#> :: A::Class2 (KDE)
#> -------------------------------------------------------------------------------- 
#> 
#> Call:
#> 	density.default(x = x, na.rm = TRUE)
#> 
#> Data: x (351 obs.);	Bandwidth 'bw' = 0.2596
#> 
#>        x                 y            
#>  Min.   :-2.5428   Min.   :4.977e-05  
#>  1st Qu.:-1.1840   1st Qu.:2.672e-02  
#>  Median : 0.1748   Median :2.239e-01  
#>  Mean   : 0.1748   Mean   :1.836e-01  
#>  3rd Qu.: 1.5336   3rd Qu.:2.926e-01  
#>  Max.   : 2.8924   Max.   :3.740e-01  
#> 
#> -------------------------------------------------------------------------------- 
#> :: B::Class1 (KDE)
#> -------------------------------------------------------------------------------- 
#> 
#> Call:
#> 	density.default(x = x, na.rm = TRUE)
#> 
#> Data: x (434 obs.);	Bandwidth 'bw' = 0.1793
#> 
#>        x                 y            
#>  Min.   :-2.4501   Min.   :5.747e-05  
#>  1st Qu.:-1.0894   1st Qu.:1.424e-02  
#>  Median : 0.2713   Median :8.798e-02  
#>  Mean   : 0.2713   Mean   :1.834e-01  
#>  3rd Qu.: 1.6320   3rd Qu.:2.758e-01  
#>  Max.   : 2.9927   Max.   :6.872e-01  
#> 
#> -------------------------------------------------------------------------------- 
#> :: B::Class2 (KDE)
#> -------------------------------------------------------------------------------- 
#> 
#> Call:
#> 	density.default(x = x, na.rm = TRUE)
#> 
#> Data: x (351 obs.);	Bandwidth 'bw' = 0.2309
#> 
#>        x                 y            
#>  Min.   :-2.4621   Min.   :5.623e-05  
#>  1st Qu.:-0.8979   1st Qu.:1.489e-02  
#>  Median : 0.6663   Median :7.738e-02  
#>  Mean   : 0.6663   Mean   :1.595e-01  
#>  3rd Qu.: 2.2305   3rd Qu.:3.336e-01  
#>  Max.   : 3.7948   Max.   :4.418e-01  
#> 
#> --------------------------------------------------------------------------------
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(naive_Bayes_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(naive_Bayes_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.249      0.751  
#> 2        0.593      0.407  
#> 3        0.332      0.668  
#> 4        0.993      0.00674
#> 5        0.978      0.0224 
#> 6        0.532      0.468
```
:::

:::

### K-Nearest Neighbors (`nearest_neighbor()`)

:::{.panel-tabset}

## `kknn`

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
nearest_neighbor_spec <- nearest_neighbor() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because kknn is the default.
  set_mode("classification")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
nearest_neighbor_fit <- nearest_neighbor_spec |>
  fit(class ~ ., data = bin_train)
nearest_neighbor_fit
#> parsnip model object
#> 
#> 
#> Call:
#> kknn::train.kknn(formula = class ~ ., data = data, ks = min_rows(5,     data, 5))
#> 
#> Type of response variable: nominal
#> Minimal misclassification: 0.2101911
#> Best kernel: optimal
#> Best k: 5
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(nearest_neighbor_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(nearest_neighbor_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1         0.2          0.8 
#> 2         0.72         0.28
#> 3         0.32         0.68
#> 4         1            0   
#> 5         1            0   
#> 6         1            0
```
:::

:::

### Null Model (`null_model()`)

:::{.panel-tabset}

## `parsnip`

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
null_model_spec <- null_model() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because parsnip is the default.
  set_mode("classification")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
null_model_fit <- null_model_spec |>
  fit(class ~ ., data = bin_train)
null_model_fit
#> parsnip model object
#> 
#> Null Regression Model
#> Predicted Value: Class1
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(null_model_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class1     
#> 2 Class1     
#> 3 Class1     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(null_model_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.553        0.447
#> 2        0.553        0.447
#> 3        0.553        0.447
#> 4        0.553        0.447
#> 5        0.553        0.447
#> 6        0.553        0.447
```
:::

:::

### Partial Least Squares (`pls()`)

:::{.panel-tabset}

## `mixOmics`

This engine requires the plsmod extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(plsmod)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
pls_spec <- pls() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because mixOmics is the default.
  set_mode("classification")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
pls_fit <- pls_spec |>
  fit(class ~ ., data = bin_train)
pls_fit
#> parsnip model object
#> 
#> 
#> Call:
#>  mixOmics::splsda(X = x, Y = y, ncomp = ncomp, keepX = keepX) 
#> 
#>  sPLS-DA (regression mode) with 2 sPLS-DA components. 
#>  You entered data X of dimensions: 785 2 
#>  You entered data Y with 2 classes. 
#> 
#>  Selection of [2] [2] variables on each of the sPLS-DA components on the X data set. 
#>  No Y variables can be selected. 
#> 
#>  Main numerical outputs: 
#>  -------------------- 
#>  loading vectors: see object$loadings 
#>  variates: see object$variates 
#>  variable names: see object$names 
#> 
#>  Functions to visualise samples: 
#>  -------------------- 
#>  plotIndiv, plotArrow, cim 
#> 
#>  Functions to visualise variables: 
#>  -------------------- 
#>  plotVar, plotLoadings, network, cim 
#> 
#>  Other functions: 
#>  -------------------- 
#>  selectVar, tune, perf, auc
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(pls_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(pls_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.462        0.538
#> 2        0.631        0.369
#> 3        0.512        0.488
#> 4        0.765        0.235
#> 5        0.675        0.325
#> 6        0.624        0.376
```
:::

:::

### Random Forests (`rand_forest()`)

:::{.panel-tabset}

## `ranger`

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
rand_forest_spec <- rand_forest() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because ranger is the default.
  set_engine("ranger", keep.inbag = TRUE) |> 
  # However, we'll set the engine and use the keep.inbag=TRUE option so that we 
  # can produce interval predictions. This is not generally required. 
  set_mode("classification")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(841)
rand_forest_fit <- rand_forest_spec |>
  fit(class ~ ., data = bin_train)
rand_forest_fit
#> parsnip model object
#> 
#> Ranger result
#> 
#> Call:
#>  ranger::ranger(x = maybe_data_frame(x), y = y, keep.inbag = ~TRUE,      num.threads = 1, verbose = FALSE, seed = sample.int(10^5,          1), probability = TRUE) 
#> 
#> Type:                             Probability estimation 
#> Number of trees:                  500 
#> Sample size:                      785 
#> Number of independent variables:  2 
#> Mtry:                             1 
#> Target node size:                 10 
#> Variable importance mode:         none 
#> Splitrule:                        gini 
#> OOB prediction error (Brier s.):  0.1477679
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(rand_forest_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(rand_forest_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.220       0.780 
#> 2        0.837       0.163 
#> 3        0.220       0.780 
#> 4        0.951       0.0485
#> 5        0.785       0.215 
#> 6        0.913       0.0868
predict(rand_forest_fit, type = "conf_int", new_data = bin_test)
#> Warning in rInfJack(x, inbag.counts): Sample size <=20, no calibration
#> performed.
#> Warning in rInfJack(x, inbag.counts): Sample size <=20, no calibration
#> performed.
#> Warning in sqrt(infjack): NaNs produced
#> # A tibble: 6 × 4
#>   .pred_lower_Class1 .pred_upper_Class1 .pred_lower_Class2 .pred_upper_Class2
#>                <dbl>              <dbl>              <dbl>              <dbl>
#> 1            0                    0.477              0.523              1    
#> 2            0.604                1                  0                  0.396
#> 3            0.01000              0.431              0.569              0.990
#> 4            0.846                1                  0                  0.154
#> 5            0.469                1                  0                  0.531
#> 6          NaN                  NaN                NaN                NaN
```
:::

## `aorsf` 

This engine requires the bonsai extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(bonsai)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
rand_forest_spec <- rand_forest() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("classification") |>
  set_engine("aorsf")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(923)
rand_forest_fit <- rand_forest_spec |>
  fit(class ~ ., data = bin_train)
rand_forest_fit
#> parsnip model object
#> 
#> ---------- Oblique random classification forest
#> 
#>      Linear combinations: Accelerated Logistic regression
#>           N observations: 785
#>                N classes: 2
#>                  N trees: 500
#>       N predictors total: 2
#>    N predictors per node: 2
#>  Average leaves per tree: 24.076
#> Min observations in leaf: 5
#>           OOB stat value: 0.87
#>            OOB stat type: AUC-ROC
#>      Variable importance: anova
#> 
#> -----------------------------------------
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(rand_forest_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(rand_forest_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.188       0.812 
#> 2        0.870       0.130 
#> 3        0.346       0.654 
#> 4        0.979       0.0206
#> 5        0.940       0.0599
#> 6        0.899       0.101
```
:::

## `grf` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
rand_forest_spec <- rand_forest() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("classification") |>
  set_engine("grf")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(546)
rand_forest_fit <- rand_forest_spec |>
  fit(class ~ ., data = bin_train)
rand_forest_fit
#> parsnip model object
#> 
#> GRF forest object of type probability_forest 
#> Number of trees: 2000 
#> Number of training samples: 785 
#> Variable importance: 
#>    1    2 
#> 0.26 0.74
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(rand_forest_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(rand_forest_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.381       0.619 
#> 2        0.779       0.221 
#> 3        0.367       0.633 
#> 4        0.981       0.0186
#> 5        0.883       0.117 
#> 6        0.797       0.203
predict(rand_forest_fit, type = "conf_int", new_data = bin_test)
#> # A tibble: 6 × 4
#>   .pred_lower_Class1 .pred_lower_Class2 .pred_upper_Class1 .pred_upper_Class2
#>                <dbl>              <dbl>              <dbl>              <dbl>
#> 1              0.567             0.806               0.194            0.433  
#> 2              0.869             0.311               0.689            0.131  
#> 3              0.585             0.852               0.148            0.415  
#> 4              1.02              0.0565              0.944           -0.0193 
#> 5              0.994             0.228               0.772            0.00601
#> 6              0.979             0.385               0.615            0.0207
```
:::

## `h2o` 

This engine requires the agua extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(agua)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
rand_forest_spec <- rand_forest() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("classification") |>
  set_engine("h2o")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(493)
rand_forest_fit <- rand_forest_spec |>
  fit(class ~ ., data = bin_train)
rand_forest_fit
#> parsnip model object
#> 
#> Model Details:
#> ==============
#> 
#> H2OBinomialModel: drf
#> Model ID:  DRF_model_R_1770287512312_5725 
#> Model Summary: 
#>   number_of_trees number_of_internal_trees model_size_in_bytes min_depth
#> 1              50                       50               92621        12
#>   max_depth mean_depth min_leaves max_leaves mean_leaves
#> 1        20   16.60000        126        166   143.08000
#> 
#> 
#> H2OBinomialMetrics: drf
#> ** Reported on training data. **
#> ** Metrics reported on Out-Of-Bag training samples **
#> 
#> MSE:  0.164699
#> RMSE:  0.4058312
#> LogLoss:  1.506369
#> Mean Per-Class Error:  0.200195
#> AUC:  0.8389854
#> AUCPR:  0.7931927
#> Gini:  0.6779708
#> R^2:  0.3337559
#> AIC:  NaN
#> 
#> Confusion Matrix (vertical: actual; across: predicted) for F1-optimal threshold:
#>        Class1 Class2    Error      Rate
#> Class1    327    107 0.246544  =107/434
#> Class2     54    297 0.153846   =54/351
#> Totals    381    404 0.205096  =161/785
#> 
#> Maximum Metrics: Maximum metrics at their respective thresholds
#>                         metric threshold      value idx
#> 1                       max f1  0.363636   0.786755 125
#> 2                       max f2  0.238095   0.832435 148
#> 3                 max f0point5  0.421053   0.760108 115
#> 4                 max accuracy  0.363636   0.794904 125
#> 5                max precision  1.000000   0.890244   0
#> 6                   max recall  0.000000   1.000000 208
#> 7              max specificity  1.000000   0.979263   0
#> 8             max absolute_mcc  0.363636   0.596505 125
#> 9   max min_per_class_accuracy  0.450000   0.785714 110
#> 10 max mean_per_class_accuracy  0.363636   0.799805 125
#> 11                     max tns  1.000000 425.000000   0
#> 12                     max fns  1.000000 278.000000   0
#> 13                     max fps  0.000000 434.000000 208
#> 14                     max tps  0.000000 351.000000 208
#> 15                     max tnr  1.000000   0.979263   0
#> 16                     max fnr  1.000000   0.792023   0
#> 17                     max fpr  0.000000   1.000000 208
#> 18                     max tpr  0.000000   1.000000 208
#> 
#> Gains/Lift Table: Extract with `h2o.gainsLift(<model>, <data>)` or `h2o.gainsLift(<model>, valid=<T/F>, xval=<T/F>)`
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(rand_forest_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(rand_forest_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.12        0.88  
#> 2        0.94        0.0600
#> 3        0.175       0.825 
#> 4        1           0     
#> 5        0.78        0.22  
#> 6        0.92        0.0800
```
:::

## `partykit` 

This engine requires the bonsai extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(bonsai)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
rand_forest_spec <- rand_forest() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("classification") |>
  set_engine("partykit")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(252)
rand_forest_fit <- rand_forest_spec |>
  fit(class ~ ., data = bin_train)
```
:::

The print method has a lot of output: 

<details>

::: {.cell layout-align="center"}

```{.r .cell-code}
capture.output(print(rand_forest_fit))[1:100] |> cat(sep = "\n")
#> parsnip model object
#> 
#> $nodes
#> $nodes[[1]]
#> [1] root
#> |   [2] V3 <= -0.06906
#> |   |   [3] V3 <= -0.61707
#> |   |   |   [4] V3 <= -0.83314
#> |   |   |   |   [5] V3 <= -0.99048
#> |   |   |   |   |   [6] V3 <= -1.29863
#> |   |   |   |   |   |   [7] V2 <= -0.93951 *
#> |   |   |   |   |   |   [8] V2 > -0.93951 *
#> |   |   |   |   |   [9] V3 > -1.29863
#> |   |   |   |   |   |   [10] V3 <= -1.21418 *
#> |   |   |   |   |   |   [11] V3 > -1.21418
#> |   |   |   |   |   |   |   [12] V2 <= -1.13676 *
#> |   |   |   |   |   |   |   [13] V2 > -1.13676
#> |   |   |   |   |   |   |   |   [14] V3 <= -1.14373 *
#> |   |   |   |   |   |   |   |   [15] V3 > -1.14373 *
#> |   |   |   |   [16] V3 > -0.99048
#> |   |   |   |   |   [17] V2 <= -1.10136 *
#> |   |   |   |   |   [18] V2 > -1.10136 *
#> |   |   |   [19] V3 > -0.83314
#> |   |   |   |   [20] V3 <= -0.68684
#> |   |   |   |   |   [21] V2 <= -0.62666 *
#> |   |   |   |   |   [22] V2 > -0.62666 *
#> |   |   |   |   [23] V3 > -0.68684 *
#> |   |   [24] V3 > -0.61707
#> |   |   |   [25] V2 <= -0.10774
#> |   |   |   |   [26] V3 <= -0.35574
#> |   |   |   |   |   [27] V3 <= -0.41085
#> |   |   |   |   |   |   [28] V3 <= -0.52674 *
#> |   |   |   |   |   |   [29] V3 > -0.52674 *
#> |   |   |   |   |   [30] V3 > -0.41085 *
#> |   |   |   |   [31] V3 > -0.35574
#> |   |   |   |   |   [32] V3 <= -0.17325 *
#> |   |   |   |   |   [33] V3 > -0.17325 *
#> |   |   |   [34] V2 > -0.10774
#> |   |   |   |   [35] V3 <= -0.38428 *
#> |   |   |   |   [36] V3 > -0.38428 *
#> |   [37] V3 > -0.06906
#> |   |   [38] V3 <= 0.54852
#> |   |   |   [39] V2 <= 0.53027
#> |   |   |   |   [40] V2 <= 0.21749
#> |   |   |   |   |   [41] V3 <= 0.09376 *
#> |   |   |   |   |   [42] V3 > 0.09376
#> |   |   |   |   |   |   [43] V3 <= 0.28687
#> |   |   |   |   |   |   |   [44] V3 <= 0.17513 *
#> |   |   |   |   |   |   |   [45] V3 > 0.17513 *
#> |   |   |   |   |   |   [46] V3 > 0.28687 *
#> |   |   |   |   [47] V2 > 0.21749 *
#> |   |   |   [48] V2 > 0.53027 *
#> |   |   [49] V3 > 0.54852
#> |   |   |   [50] V2 <= 1.99786
#> |   |   |   |   [51] V3 <= 1.02092
#> |   |   |   |   |   [52] V2 <= 0.5469
#> |   |   |   |   |   |   [53] V3 <= 0.83487
#> |   |   |   |   |   |   |   [54] V2 <= 0.36626 *
#> |   |   |   |   |   |   |   [55] V2 > 0.36626 *
#> |   |   |   |   |   |   [56] V3 > 0.83487 *
#> |   |   |   |   |   [57] V2 > 0.5469
#> |   |   |   |   |   |   [58] V3 <= 0.62673 *
#> |   |   |   |   |   |   [59] V3 > 0.62673 *
#> |   |   |   |   [60] V3 > 1.02092
#> |   |   |   |   |   [61] V3 <= 1.29539
#> |   |   |   |   |   |   [62] V3 <= 1.2241 *
#> |   |   |   |   |   |   [63] V3 > 1.2241 *
#> |   |   |   |   |   [64] V3 > 1.29539
#> |   |   |   |   |   |   [65] V3 <= 2.01809 *
#> |   |   |   |   |   |   [66] V3 > 2.01809 *
#> |   |   |   [67] V2 > 1.99786 *
#> 
#> $nodes[[2]]
#> [1] root
#> |   [2] V3 <= -0.00054
#> |   |   [3] V3 <= -0.58754
#> |   |   |   [4] V3 <= -0.83314
#> |   |   |   |   [5] V2 <= -1.15852
#> |   |   |   |   |   [6] V2 <= -1.76192 *
#> |   |   |   |   |   [7] V2 > -1.76192 *
#> |   |   |   |   [8] V2 > -1.15852
#> |   |   |   |   |   [9] V3 <= -1.21418
#> |   |   |   |   |   |   [10] V3 <= -1.32176 *
#> |   |   |   |   |   |   [11] V3 > -1.32176 *
#> |   |   |   |   |   [12] V3 > -1.21418
#> |   |   |   |   |   |   [13] V2 <= -1.08164 *
#> |   |   |   |   |   |   [14] V2 > -1.08164
#> |   |   |   |   |   |   |   [15] V3 <= -1.14373 *
#> |   |   |   |   |   |   |   [16] V3 > -1.14373 *
#> |   |   |   [17] V3 > -0.83314
#> |   |   |   |   [18] V2 <= -0.51524
#> |   |   |   |   |   [19] V3 <= -0.66041
#> |   |   |   |   |   |   [20] V3 <= -0.70885 *
#> |   |   |   |   |   |   [21] V3 > -0.70885 *
#> |   |   |   |   |   [22] V3 > -0.66041 *
#> |   |   |   |   [23] V2 > -0.51524 *
#> |   |   [24] V3 > -0.58754
#> |   |   |   [25] V2 <= -0.07243
#> |   |   |   |   [26] V3 <= -0.31247
#> |   |   |   |   |   [27] V2 <= -0.98014 *
```
:::

</details>

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(rand_forest_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(rand_forest_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.375       0.625 
#> 2        0.813       0.187 
#> 3        0.284       0.716 
#> 4        0.963       0.0365
#> 5        0.892       0.108 
#> 6        0.922       0.0785
```
:::

## `randomForest` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
rand_forest_spec <- rand_forest() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("classification") |>
  set_engine("randomForest")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(726)
rand_forest_fit <- rand_forest_spec |>
  fit(class ~ ., data = bin_train)
rand_forest_fit
#> parsnip model object
#> 
#> 
#> Call:
#>  randomForest(x = maybe_data_frame(x), y = y) 
#>                Type of random forest: classification
#>                      Number of trees: 500
#> No. of variables tried at each split: 1
#> 
#>         OOB estimate of  error rate: 21.53%
#> Confusion matrix:
#>        Class1 Class2 class.error
#> Class1    349     85   0.1958525
#> Class2     84    267   0.2393162
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(rand_forest_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(rand_forest_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.162        0.838
#> 2        0.848        0.152
#> 3        0.108        0.892
#> 4        1            0    
#> 5        0.74         0.26 
#> 6        0.91         0.09
```
:::

## `spark` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
rand_forest_spec <- rand_forest() |>
  set_mode("classification") |>
  set_engine("spark")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(693)
rand_forest_fit <- rand_forest_spec |>
  fit(Class ~ ., data = tbl_bin$training)
rand_forest_fit
#> parsnip model object
#> 
#> Formula: Class ~ .
#> 
#> RandomForestClassificationModel: uid=random_forest__c160ba73_c71f_46ca_9bf4_22a163b941e4, numTrees=20, numClasses=2, numFeatures=2
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(rand_forest_fit, type = "class", new_data = tbl_bin$test)
#> # Source:   SQL [?? x 1]
#> # Database: spark_connection
#>   pred_class
#>   <chr>     
#> 1 Class2    
#> 2 Class2    
#> 3 Class1    
#> 4 Class2    
#> 5 Class2    
#> 6 Class1    
#> 7 Class2
predict(rand_forest_fit, type = "prob", new_data = tbl_bin$test)
#> # Source:   SQL [?? x 2]
#> # Database: spark_connection
#>   pred_Class1 pred_Class2
#>         <dbl>       <dbl>
#> 1      0.315       0.685 
#> 2      0.241       0.759 
#> 3      0.732       0.268 
#> 4      0.235       0.765 
#> 5      0.259       0.741 
#> 6      0.933       0.0674
#> 7      0.0968      0.903
```
:::

:::

### Rule Fit (`rule_fit()`)

:::{.panel-tabset}

## `xrf`

This engine requires the rules extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(rules)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
rule_fit_spec <- rule_fit() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because xrf is the default.
  set_mode("classification")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(95)
rule_fit_fit <- rule_fit_spec |>
  fit(class ~ ., data = bin_train)
rule_fit_fit
#> parsnip model object
#> 
#> An eXtreme RuleFit model of 368 rules.
#> 
#> Original Formula:
#> 
#> class ~ A + B
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(rule_fit_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class1     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(rule_fit_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.413        0.587
#> 2        0.641        0.359
#> 3        0.530        0.470
#> 4        0.890        0.110
#> 5        0.804        0.196
#> 6        0.607        0.393
```
:::

## `h2o` 

This engine requires the agua extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(agua)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
rule_fit_spec <- rule_fit() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("classification") |>
  set_engine("h2o")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(536)
rule_fit_fit <- rule_fit_spec |>
  fit(class ~ ., data = bin_train)
rule_fit_fit
#> parsnip model object
#> 
#> Model Details:
#> ==============
#> 
#> H2OBinomialModel: rulefit
#> Model ID:  RuleFit_model_R_1770287512312_5776 
#> Rulefit Model Summary: 
#>     family  link            regularization number_of_predictors_total
#> 1 binomial logit Lasso (lambda = 0.03081 )                       2329
#>   number_of_active_predictors number_of_iterations rule_ensemble_size
#> 1                           3                    4               2327
#>   number_of_trees number_of_internal_trees min_depth max_depth mean_depth
#> 1             150                      150         0         5    4.00000
#>   min_leaves max_leaves mean_leaves
#> 1          0         29    15.51333
#> 
#> 
#> H2OBinomialMetrics: rulefit
#> ** Reported on training data. **
#> 
#> MSE:  0.1411478
#> RMSE:  0.3756964
#> LogLoss:  0.4472749
#> Mean Per-Class Error:  0.1850933
#> AUC:  0.8779327
#> AUCPR:  0.8372496
#> Gini:  0.7558654
#> 
#> Confusion Matrix (vertical: actual; across: predicted) for F1-optimal threshold:
#>        Class1 Class2    Error      Rate
#> Class1    350     84 0.193548   =84/434
#> Class2     62    289 0.176638   =62/351
#> Totals    412    373 0.185987  =146/785
#> 
#> Maximum Metrics: Maximum metrics at their respective thresholds
#>                         metric threshold      value idx
#> 1                       max f1  0.499611   0.798343 199
#> 2                       max f2  0.226927   0.861169 285
#> 3                 max f0point5  0.626200   0.803634 144
#> 4                 max accuracy  0.523044   0.815287 191
#> 5                max precision  0.980574   1.000000   0
#> 6                   max recall  0.052101   1.000000 394
#> 7              max specificity  0.980574   1.000000   0
#> 8             max absolute_mcc  0.523044   0.627478 191
#> 9   max min_per_class_accuracy  0.512020   0.813364 196
#> 10 max mean_per_class_accuracy  0.499611   0.814907 199
#> 11                     max tns  0.980574 434.000000   0
#> 12                     max fns  0.980574 350.000000   0
#> 13                     max fps  0.043433 434.000000 399
#> 14                     max tps  0.052101 351.000000 394
#> 15                     max tnr  0.980574   1.000000   0
#> 16                     max fnr  0.980574   0.997151   0
#> 17                     max fpr  0.043433   1.000000 399
#> 18                     max tpr  0.052101   1.000000 394
#> 
#> Gains/Lift Table: Extract with `h2o.gainsLift(<model>, <data>)` or `h2o.gainsLift(<model>, valid=<T/F>, xval=<T/F>)`
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(rule_fit_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(rule_fit_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.393       0.607 
#> 2        0.739       0.261 
#> 3        0.455       0.545 
#> 4        0.956       0.0442
#> 5        0.882       0.118 
#> 6        0.693       0.307
```
:::

:::

### Support Vector Machine (Linear Kernel) (`svm_linear()`)

:::{.panel-tabset}

## `kernlab`

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
svm_linear_spec <- svm_linear() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("classification") |>
  set_engine("kernlab")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
svm_linear_fit <- svm_linear_spec |>
  fit(class ~ ., data = bin_train)
svm_linear_fit
#> parsnip model object
#> 
#> Support Vector Machine object of class "ksvm" 
#> 
#> SV type: C-svc  (classification) 
#>  parameter : cost C = 1 
#> 
#> Linear (vanilla) kernel function. 
#> 
#> Number of Support Vectors : 357 
#> 
#> Objective Function Value : -353.0043 
#> Training error : 0.17707 
#> Probability model included.
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(svm_linear_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class1     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(svm_linear_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.404       0.596 
#> 2        0.858       0.142 
#> 3        0.541       0.459 
#> 4        0.975       0.0254
#> 5        0.905       0.0950
#> 6        0.850       0.150
```
:::

## `LiblineaR` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
svm_linear_spec <- svm_linear() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because LiblineaR is the default.
  set_mode("classification")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
svm_linear_fit <- svm_linear_spec |>
  fit(class ~ ., data = bin_train)
svm_linear_fit
#> parsnip model object
#> 
#> $TypeDetail
#> [1] "L2-regularized L2-loss support vector classification dual (L2R_L2LOSS_SVC_DUAL)"
#> 
#> $Type
#> [1] 1
#> 
#> $W
#>              A          B      Bias
#> [1,] 0.3641766 -0.9648797 0.1182725
#> 
#> $Bias
#> [1] 1
#> 
#> $ClassNames
#> [1] Class1 Class2
#> Levels: Class1 Class2
#> 
#> $NbClass
#> [1] 2
#> 
#> attr(,"class")
#> [1] "LiblineaR"
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(svm_linear_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class1     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
```
:::

:::

### Support Vector Machine (Polynomial Kernel) (`svm_poly()`)

:::{.panel-tabset}

## `kernlab`

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
svm_poly_spec <- svm_poly() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because kernlab is the default.
  set_mode("classification")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
svm_poly_fit <- svm_poly_spec |>
  fit(class ~ ., data = bin_train)
#>  Setting default kernel parameters
svm_poly_fit
#> parsnip model object
#> 
#> Support Vector Machine object of class "ksvm" 
#> 
#> SV type: C-svc  (classification) 
#>  parameter : cost C = 1 
#> 
#> Polynomial kernel function. 
#>  Hyperparameters : degree =  1  scale =  1  offset =  1 
#> 
#> Number of Support Vectors : 357 
#> 
#> Objective Function Value : -353.0043 
#> Training error : 0.17707 
#> Probability model included.
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(svm_poly_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class2     
#> 2 Class1     
#> 3 Class1     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(svm_poly_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.399       0.601 
#> 2        0.861       0.139 
#> 3        0.538       0.462 
#> 4        0.976       0.0237
#> 5        0.908       0.0917
#> 6        0.853       0.147
```
:::

:::

### Support Vector Machine (Radial Basis Function Kernel) (`svm_rbf()`)

:::{.panel-tabset}

## `kernlab`

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
svm_rbf_spec <- svm_rbf() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because kernlab is the default.
  set_mode("classification")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
svm_rbf_fit <- svm_rbf_spec |>
  fit(class ~ ., data = bin_train)
svm_rbf_fit
#> parsnip model object
#> 
#> Support Vector Machine object of class "ksvm" 
#> 
#> SV type: C-svc  (classification) 
#>  parameter : cost C = 1 
#> 
#> Gaussian Radial Basis kernel function. 
#>  Hyperparameter : sigma =  1.9107071282545 
#> 
#> Number of Support Vectors : 335 
#> 
#> Objective Function Value : -296.4885 
#> Training error : 0.173248 
#> Probability model included.
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(svm_rbf_fit, type = "class", new_data = bin_test)
#> # A tibble: 6 × 1
#>   .pred_class
#>   <fct>      
#> 1 Class1     
#> 2 Class1     
#> 3 Class2     
#> 4 Class1     
#> 5 Class1     
#> 6 Class1
predict(svm_rbf_fit, type = "prob", new_data = bin_test)
#> # A tibble: 6 × 2
#>   .pred_Class1 .pred_Class2
#>          <dbl>        <dbl>
#> 1        0.547        0.453
#> 2        0.871        0.129
#> 3        0.260        0.740
#> 4        0.861        0.139
#> 5        0.863        0.137
#> 6        0.863        0.137
```
:::

:::

# Regression Models

## Example data

To demonstrate regression, we'll subset some data, make a training/test split, and standardize the predictors: 

::: {.cell layout-align="center"}

```{.r .cell-code}
set.seed(938)
reg_split <-
  modeldata::concrete |> 
  slice_sample(n = 100) |> 
  select(strength = compressive_strength, cement, age) |> 
  initial_split(prop = 0.95, strata = strength)
reg_split
#> <Training/Testing/Total>
#> <92/8/100>

reg_rec <- 
  recipe(strength ~ ., data = training(reg_split)) |> 
  step_normalize(all_numeric_predictors()) |> 
  prep()

reg_train <- bake(reg_rec, new_data = NULL)
reg_test <- bake(reg_rec, new_data = testing(reg_split))
```
:::

We also have models that are specifically designed for integer count outcomes. The data for these are:

::: {.cell layout-align="center"}

```{.r .cell-code}
set.seed(207)
count_split <-
  attrition |>
  select(num_years = TotalWorkingYears, age = Age, income = MonthlyIncome) |>
  initial_split(prop = 0.994)
count_split
#> <Training/Testing/Total>
#> <1461/9/1470>

count_rec <-
  recipe(num_years ~ ., data = training(count_split)) |>
  step_normalize(all_numeric_predictors()) |>
  prep()

count_train <- bake(count_rec, new_data = NULL)
count_test <- bake(count_rec, new_data = testing(count_split))
```
:::

Finally, we have some models that handle hierarchical data, where some rows are statistically correlated with other rows. For these examples, we'll use a data set that models body weights as a function of time for several "subjects" (rats, actually). We'll split these data in a way where all rows for a specific subject are either in the training or the test set: 

::: {.cell layout-align="center"}

```{.r .cell-code}
set.seed(224)
reg_group_split <- 
  nlme::BodyWeight |> 
  # Get rid of some extra attributes added by the nlme package
  as_tibble() |> 
  # Convert to an _unordered_ factor
  mutate(Rat = factor(as.character(Rat))) |> 
  group_initial_split(group = Rat)
reg_group_train <- training(reg_group_split)
reg_group_test <- testing(reg_group_split)
```
:::

There are 12 subjects in the training set and 4 in the test set. 

If using the **Apache Spark** engine, we will need to identify the data source, and then use it to create the splits. For this article, we will copy the `concrete` data set into the Spark session.

::: {.cell layout-align="center"}

```{.r .cell-code}
library(sparklyr)
sc <- spark_connect("local")
#> Re-using existing Spark connection to local

tbl_concrete <- copy_to(sc, modeldata::concrete)

tbl_reg <- sdf_random_split(tbl_concrete, training = 0.95, test = 0.05, seed = 100)
```
:::

## Models

### Bagged MARS (`bag_mars()`)

:::{.panel-tabset}

## `earth`

This engine requires the baguette extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(baguette)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
bag_mars_spec <- bag_mars() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because earth is the default.
  set_mode("regression")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(147)
bag_mars_fit <- bag_mars_spec |> 
  fit(strength ~ ., data = reg_train)
bag_mars_fit
#> parsnip model object
#> 
#> Bagged MARS (regression with 11 members)
#> 
#> Variable importance scores include:
#> 
#> # A tibble: 2 × 4
#>   term   value std.error  used
#>   <chr>  <dbl>     <dbl> <int>
#> 1 age     93.1      4.61    11
#> 2 cement  69.4      4.95    11
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(bag_mars_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  22.4
#> 2  41.9
#> 3  26.7
#> 4  56.6
#> 5  36.4
#> 6  36.2
#> 7  37.8
#> 8  37.7
```
:::

:::

### Bagged Neural Networks (`bag_mlp()`)

:::{.panel-tabset}

## `nnet`

This engine requires the baguette extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(baguette)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
bag_mlp_spec <- bag_mlp() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because nnet is the default.
  set_mode("regression")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(324)
bag_mlp_fit <- bag_mlp_spec |>
  fit(strength ~ ., data = reg_train)
bag_mlp_fit
#> parsnip model object
#> 
#> Bagged nnet (regression with 11 members)
#> 
#> Variable importance scores include:
#> 
#> # A tibble: 2 × 4
#>   term   value std.error  used
#>   <chr>  <dbl>     <dbl> <int>
#> 1 age     55.9      2.96    11
#> 2 cement  44.1      2.96    11
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(bag_mlp_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  19.9
#> 2  39.1
#> 3  28.3
#> 4  68.8
#> 5  44.1
#> 6  36.3
#> 7  40.8
#> 8  37.0
```
:::

:::

### Bagged Decision Trees (`bag_tree()`)

:::{.panel-tabset}

## `rpart`

This engine requires the baguette extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(baguette)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
bag_tree_spec <- bag_tree() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because rpart is the default.
  set_mode("regression")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(230)
bag_tree_fit <- bag_tree_spec |>
  fit(strength ~ ., data = reg_train)
bag_tree_fit
#> parsnip model object
#> 
#> Bagged CART (regression with 11 members)
#> 
#> Variable importance scores include:
#> 
#> # A tibble: 2 × 4
#>   term    value std.error  used
#>   <chr>   <dbl>     <dbl> <int>
#> 1 cement 16621.     1392.    11
#> 2 age    12264.      710.    11
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(bag_tree_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  23.0
#> 2  33.0
#> 3  29.6
#> 4  54.2
#> 5  36.2
#> 6  39.4
#> 7  40.7
#> 8  46.5
```
:::

:::

### Bayesian Additive Regression Trees (`bart()`)

:::{.panel-tabset}

## `dbarts`

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
bart_spec <- bart() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because dbarts is the default.
  set_mode("regression")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(134)
bart_fit <- bart_spec |>
  fit(strength ~ ., data = reg_train)
bart_fit
#> parsnip model object
#> 
#> 
#> Call:
#> `NULL`()
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(bart_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  24.2
#> 2  40.9
#> 3  26.0
#> 4  52.0
#> 5  36.5
#> 6  36.7
#> 7  39.0
#> 8  37.8
predict(bart_fit, type = "conf_int", new_data = reg_test)
#> # A tibble: 8 × 2
#>   .pred_lower .pred_upper
#>         <dbl>       <dbl>
#> 1        17.0        32.4
#> 2        33.0        48.9
#> 3        20.1        31.5
#> 4        42.0        62.5
#> 5        28.5        44.5
#> 6        30.3        42.3
#> 7        33.1        45.3
#> 8        26.3        48.8
predict(bart_fit, type = "pred_int", new_data = reg_test)
#> # A tibble: 8 × 2
#>   .pred_lower .pred_upper
#>         <dbl>       <dbl>
#> 1        5.00        41.8
#> 2       19.9         60.5
#> 3        7.37        44.3
#> 4       32.4         72.1
#> 5       15.7         56.4
#> 6       18.9         56.8
#> 7       21.2         57.2
#> 8       17.2         58.5
```
:::

:::

### Boosted Decision Trees (`boost_tree()`)

:::{.panel-tabset}

## `xgboost`

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
boost_tree_spec <- boost_tree() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because xgboost is the default.
  set_mode("regression")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(748)
boost_tree_fit <- boost_tree_spec |>
  fit(strength ~ ., data = reg_train)
boost_tree_fit
#> parsnip model object
#> 
#> ##### xgb.Booster
#> call:
#>   xgboost::xgb.train(params = list(eta = 0.3, max_depth = 6, gamma = 0, 
#>     colsample_bytree = 1, colsample_bynode = 1, min_child_weight = 1, 
#>     subsample = 1, nthread = 1, objective = "reg:squarederror"), 
#>     data = x$data, nrounds = 15, evals = x$watchlist, verbose = 0)
#> # of features: 2 
#> # of rounds:  15 
#> callbacks:
#>    evaluation_log 
#> evaluation_log:
#>   iter training_rmse
#>  <num>         <num>
#>      1     13.566385
#>      2     10.756125
#>    ---           ---
#>     14      2.269873
#>     15      2.118914
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(boost_tree_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  26.9
#> 2  32.8
#> 3  27.9
#> 4  55.3
#> 5  35.5
#> 6  37.4
#> 7  41.1
#> 8  33.5
```
:::

## `catboost` 

This engine requires the bonsai extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(bonsai)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
boost_tree_spec <- boost_tree() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("regression") |>
  set_engine("catboost")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(557)
boost_tree_fit <- boost_tree_spec |>
  fit(strength ~ ., data = reg_train)
boost_tree_fit
#> parsnip model object
#> 
#> CatBoost model (1000 trees)
#> Loss function: RMSE
#> Fit to 2 feature(s)
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(boost_tree_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  26.6
#> 2  33.9
#> 3  27.8
#> 4  60.6
#> 5  34.7
#> 6  36.3
#> 7  43.6
#> 8  29.3
```
:::

## `h2o` 

This engine requires the agua extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(agua)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
boost_tree_spec <- boost_tree() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("regression") |>
  set_engine("h2o_gbm")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(720)
boost_tree_fit <- boost_tree_spec |>
  fit(strength ~ ., data = reg_train)
boost_tree_fit
#> parsnip model object
#> 
#> Model Details:
#> ==============
#> 
#> H2ORegressionModel: gbm
#> Model ID:  GBM_model_R_1770287512312_5932 
#> Model Summary: 
#>   number_of_trees number_of_internal_trees model_size_in_bytes min_depth
#> 1              50                       50               20474         6
#>   max_depth mean_depth min_leaves max_leaves mean_leaves
#> 1         6    6.00000         14         43    27.92000
#> 
#> 
#> H2ORegressionMetrics: gbm
#> ** Reported on training data. **
#> 
#> MSE:  0.001563879
#> RMSE:  0.03954591
#> MAE:  0.02903684
#> RMSLE:  0.001771464
#> Mean Residual Deviance :  0.001563879
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(boost_tree_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  29.7
#> 2  32.2
#> 3  26.9
#> 4  63.2
#> 5  34.9
#> 6  39.0
#> 7  40.0
#> 8  32.9
```
:::

## `h2o_gbm` 

This engine requires the agua extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(agua)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
boost_tree_spec <- boost_tree() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("regression") |>
  set_engine("h2o_gbm")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(90)
boost_tree_fit <- boost_tree_spec |>
  fit(strength ~ ., data = reg_train)
boost_tree_fit
#> parsnip model object
#> 
#> Model Details:
#> ==============
#> 
#> H2ORegressionModel: gbm
#> Model ID:  GBM_model_R_1770287512312_5933 
#> Model Summary: 
#>   number_of_trees number_of_internal_trees model_size_in_bytes min_depth
#> 1              50                       50               20474         6
#>   max_depth mean_depth min_leaves max_leaves mean_leaves
#> 1         6    6.00000         14         43    27.92000
#> 
#> 
#> H2ORegressionMetrics: gbm
#> ** Reported on training data. **
#> 
#> MSE:  0.001563879
#> RMSE:  0.03954591
#> MAE:  0.02903684
#> RMSLE:  0.001771464
#> Mean Residual Deviance :  0.001563879
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(boost_tree_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  29.7
#> 2  32.2
#> 3  26.9
#> 4  63.2
#> 5  34.9
#> 6  39.0
#> 7  40.0
#> 8  32.9
```
:::

## `lightgbm` 

This engine requires the bonsai extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(bonsai)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
boost_tree_spec <- boost_tree() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("regression") |>
  set_engine("lightgbm")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(570)
boost_tree_fit <- boost_tree_spec |>
  fit(strength ~ ., data = reg_train)
boost_tree_fit
#> parsnip model object
#> 
#> LightGBM Model (100 trees)
#> Objective: regression
#> Fitted to dataset with 2 columns
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(boost_tree_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  20.6
#> 2  42.5
#> 3  27.0
#> 4  49.2
#> 5  43.7
#> 6  38.3
#> 7  41.1
#> 8  36.9
```
:::

## `spark` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
boost_tree_spec <- boost_tree() |>
  set_mode("regression") |>
  set_engine("spark")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(620)
boost_tree_fit <- boost_tree_spec |>
  fit(compressive_strength ~ ., data = tbl_reg$training)
boost_tree_fit
#> parsnip model object
#> 
#> Formula: compressive_strength ~ .
#> 
#> GBTRegressionModel: uid=gradient_boosted_trees__5897a9f5_9ed9_4360_80cb_7c5694d8b78b, numTrees=20, numFeatures=8
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(boost_tree_fit, new_data = tbl_reg$test)
#> # Source:   SQL [?? x 1]
#> # Database: spark_connection
#>     pred
#>    <dbl>
#>  1 20.8 
#>  2 28.1 
#>  3 15.5 
#>  4 22.4 
#>  5  9.37
#>  6 40.1 
#>  7 14.2 
#>  8 32.1 
#>  9 37.4 
#> 10 49.5 
#> # ℹ more rows
```
:::

:::

### Cubist Rules (`cubist_rules()`)

:::{.panel-tabset}

## `Cubist` 

This engine requires the rules extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(rules)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
# This model only works with a single mode, so we don't need to set the mode.
# We don't need to set the engine because Cubist is the default.
cubist_rules_spec <- cubist_rules()
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(188)
cubist_rules_fit <- cubist_rules_spec |>
  fit(strength ~ ., data = reg_train)
cubist_rules_fit
#> parsnip model object
#> 
#> 
#> Call:
#> cubist.default(x = x, y = y, committees = 1)
#> 
#> Number of samples: 92 
#> Number of predictors: 2 
#> 
#> Number of committees: 1 
#> Number of rules: 2
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(cubist_rules_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  24.2
#> 2  46.3
#> 3  23.6
#> 4  54.4
#> 5  32.7
#> 6  37.8
#> 7  38.8
#> 8  38.6
```
:::

:::

### Decision Tree (`decision_tree()`)

:::{.panel-tabset}

## `rpart`

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
decision_tree_spec <- decision_tree() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because rpart is the default.
  set_mode("regression")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
decision_tree_fit <- decision_tree_spec |>
  fit(strength ~ ., data = reg_train)
decision_tree_fit
#> parsnip model object
#> 
#> n= 92 
#> 
#> node), split, n, deviance, yval
#>       * denotes terminal node
#> 
#>  1) root 92 26564.7400 33.57728  
#>    2) cement< 0.7861846 69 12009.9000 27.81493  
#>      4) age< -0.5419541 23   964.6417 14.42348  
#>        8) cement< -0.3695209 12   292.7811 11.14083 *
#>        9) cement>=-0.3695209 11   401.4871 18.00455 *
#>      5) age>=-0.5419541 46  4858.3440 34.51065  
#>       10) age< 0.008934354 32  2208.3040 31.16781  
#>         20) cement< 0.311975 24  1450.6200 28.75583 *
#>         21) cement>=0.311975 8   199.1900 38.40375 *
#>       11) age>=0.008934354 14  1475.1130 42.15143 *
#>    3) cement>=0.7861846 23  5390.3320 50.86435  
#>      6) age< -0.5419541 7   390.4204 40.08429 *
#>      7) age>=-0.5419541 16  3830.5510 55.58062 *
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(decision_tree_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  18.0
#> 2  42.2
#> 3  28.8
#> 4  55.6
#> 5  40.1
#> 6  38.4
#> 7  38.4
#> 8  40.1
```
:::

## `partykit` 

This engine requires the bonsai extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(bonsai)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
decision_tree_spec <- decision_tree() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("regression") |>
  set_engine("partykit")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
decision_tree_fit <- decision_tree_spec |>
  fit(strength ~ ., data = reg_train)
decision_tree_fit
#> parsnip model object
#> 
#> 
#> Model formula:
#> strength ~ cement + age
#> 
#> Fitted party:
#> [1] root
#> |   [2] cement <= 0.72078
#> |   |   [3] age <= -0.60316
#> |   |   |   [4] cement <= -0.38732: 11.141 (n = 12, err = 292.8)
#> |   |   |   [5] cement > -0.38732: 18.005 (n = 11, err = 401.5)
#> |   |   [6] age > -0.60316
#> |   |   |   [7] cement <= 0.24945
#> |   |   |   |   [8] age <= -0.2359: 28.756 (n = 24, err = 1450.6)
#> |   |   |   |   [9] age > -0.2359: 39.014 (n = 11, err = 634.8)
#> |   |   |   [10] cement > 0.24945: 42.564 (n = 11, err = 1041.7)
#> |   [11] cement > 0.72078: 50.864 (n = 23, err = 5390.3)
#> 
#> Number of inner nodes:    5
#> Number of terminal nodes: 6
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(decision_tree_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  18.0
#> 2  39.0
#> 3  28.8
#> 4  50.9
#> 5  50.9
#> 6  42.6
#> 7  42.6
#> 8  50.9
```
:::

## `spark` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
decision_tree_spec <- decision_tree() |>
  set_mode("regression") |> 
  set_engine("spark")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
decision_tree_fit <- decision_tree_spec |>
  fit(compressive_strength ~ ., data = tbl_reg$training)
decision_tree_fit
#> parsnip model object
#> 
#> Formula: compressive_strength ~ .
#> 
#> DecisionTreeRegressionModel: uid=decision_tree_regressor__b9233b83_d74a_41e1_97df_0e3d97b8556e, depth=5, numNodes=63, numFeatures=8
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(decision_tree_fit, new_data = tbl_reg$test)
#> # Source:   SQL [?? x 1]
#> # Database: spark_connection
#>     pred
#>    <dbl>
#>  1  26.7
#>  2  26.7
#>  3  14.9
#>  4  26.7
#>  5  10.5
#>  6  40.2
#>  7  15.0
#>  8  40.2
#>  9  40.2
#> 10  41.4
#> # ℹ more rows
```
:::

:::

### Generalized Additive Models (`gen_additive_mod()`)

:::{.panel-tabset}

## `mgcv`

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
gen_additive_mod_spec <- gen_additive_mod() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because mgcv is the default.
  set_mode("regression")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
gen_additive_mod_fit <- 
  gen_additive_mod_spec |> 
  fit(strength ~ s(age) + s(cement), data = reg_train)
gen_additive_mod_fit
#> parsnip model object
#> 
#> 
#> Family: gaussian 
#> Link function: identity 
#> 
#> Formula:
#> strength ~ s(age) + s(cement)
#> 
#> Estimated degrees of freedom:
#> 4.18 3.56  total = 8.74 
#> 
#> GCV score: 108.4401
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(gen_additive_mod_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  23.1
#> 2  41.2
#> 3  26.7
#> 4  55.9
#> 5  35.2
#> 6  37.1
#> 7  38.5
#> 8  39.6
predict(gen_additive_mod_fit, type = "conf_int", new_data = reg_test)
#> # A tibble: 8 × 2
#>   .pred_lower .pred_upper
#>     <dbl[1d]>   <dbl[1d]>
#> 1        18.9        27.4
#> 2        35.7        46.6
#> 3        22.4        31.0
#> 4        47.0        64.7
#> 5        30.1        40.4
#> 6        32.9        41.2
#> 7        34.3        42.6
#> 8        30.3        49.0
```
:::

:::

### Linear Reg (`linear_reg()`)

:::{.panel-tabset}

## `lm` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
# This model only works with a single mode, so we don't need to set the mode.
# We don't need to set the engine because lm is the default.
linear_reg_spec <- linear_reg()
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_fit <- linear_reg_spec |>
  fit(strength ~ ., data = reg_train)
linear_reg_fit
#> parsnip model object
#> 
#> 
#> Call:
#> stats::lm(formula = strength ~ ., data = data)
#> 
#> Coefficients:
#> (Intercept)       cement          age  
#>      33.577        8.795        5.471
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(linear_reg_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  32.1
#> 2  30.3
#> 3  21.6
#> 4  51.4
#> 5  40.3
#> 6  35.3
#> 7  36.3
#> 8  48.8
predict(linear_reg_fit, type = "conf_int", new_data = reg_test)
#> # A tibble: 8 × 2
#>   .pred_lower .pred_upper
#>         <dbl>       <dbl>
#> 1        28.8        35.4
#> 2        27.1        33.5
#> 3        17.3        25.9
#> 4        44.6        58.1
#> 5        35.6        45.0
#> 6        32.3        38.3
#> 7        33.2        39.4
#> 8        41.6        56.0
predict(linear_reg_fit, type = "pred_int", new_data = reg_test)
#> # A tibble: 8 × 2
#>   .pred_lower .pred_upper
#>         <dbl>       <dbl>
#> 1        5.72        58.5
#> 2        3.89        56.7
#> 3       -4.94        48.2
#> 4       24.3         78.5
#> 5       13.7         67.0
#> 6        8.95        61.7
#> 7        9.89        62.7
#> 8       21.6         76.0
```
:::

## `brulee` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_spec <- linear_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("brulee")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(1)
linear_reg_fit <- linear_reg_spec |>
  fit(strength ~ ., data = reg_train)
linear_reg_fit
#> parsnip model object
#> 
#> Linear regression
#> 
#> 92 samples, 2 features, numeric outcome 
#> weight decay: 0.001 
#> batch size: 83 
#> scaled validation loss after 1 epoch: 235
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(linear_reg_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  32.1
#> 2  30.1
#> 3  21.6
#> 4  51.2
#> 5  40.3
#> 6  35.2
#> 7  36.2
#> 8  48.7
```
:::

## `gee` 

This engine requires the multilevelmod extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(multilevelmod)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_spec <- linear_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("gee")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_fit <- 
  linear_reg_spec |> 
  fit(weight ~ Time + Diet + id_var(Rat), data = reg_group_train)
#> Beginning Cgee S-function, @(#) geeformula.q 4.13 98/01/27
#> running glm to get initial regression estimate
linear_reg_fit
#> parsnip model object
#> 
#> 
#>  GEE:  GENERALIZED LINEAR MODELS FOR DEPENDENT DATA
#>  gee S-function, version 4.13 modified 98/01/27 (1998) 
#> 
#> Model:
#>  Link:                      Identity 
#>  Variance to Mean Relation: Gaussian 
#>  Correlation Structure:     Independent 
#> 
#> Call:
#> gee::gee(formula = weight ~ Time + Diet, id = data$Rat, data = data, 
#>     family = gaussian)
#> 
#> Number of observations :  132 
#> 
#> Maximum cluster size   :  11 
#> 
#> 
#> Coefficients:
#> (Intercept)        Time       Diet2       Diet3 
#>  245.410439    0.549192  185.621212  259.287879 
#> 
#> Estimated Scale Parameter:  272.1604
#> Number of Iterations:  1
#> 
#> Working Correlation[1:4,1:4]
#>      [,1] [,2] [,3] [,4]
#> [1,]    1    0    0    0
#> [2,]    0    1    0    0
#> [3,]    0    0    1    0
#> [4,]    0    0    0    1
#> 
#> 
#> Returned Error Value:
#> [1] 0
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(linear_reg_fit, new_data = reg_group_test)
#> # A tibble: 44 × 1
#>    .pred
#>    <dbl>
#>  1  246.
#>  2  250.
#>  3  254.
#>  4  257.
#>  5  261.
#>  6  265.
#>  7  269.
#>  8  270.
#>  9  273.
#> 10  277.
#> # ℹ 34 more rows
```
:::

## `glm` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_spec <- linear_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("glm")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_fit <- linear_reg_spec |>
  fit(strength ~ ., data = reg_train)
linear_reg_fit
#> parsnip model object
#> 
#> 
#> Call:  stats::glm(formula = strength ~ ., family = stats::gaussian, 
#>     data = data)
#> 
#> Coefficients:
#> (Intercept)       cement          age  
#>      33.577        8.795        5.471  
#> 
#> Degrees of Freedom: 91 Total (i.e. Null);  89 Residual
#> Null Deviance:	    26560 
#> Residual Deviance: 15480 	AIC: 740.6
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(linear_reg_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  32.1
#> 2  30.3
#> 3  21.6
#> 4  51.4
#> 5  40.3
#> 6  35.3
#> 7  36.3
#> 8  48.8
predict(linear_reg_fit, type = "conf_int", new_data = reg_test)
#> # A tibble: 8 × 2
#>   .pred_lower .pred_upper
#>         <dbl>       <dbl>
#> 1        28.8        35.4
#> 2        27.1        33.5
#> 3        17.3        25.9
#> 4        44.6        58.1
#> 5        35.6        45.0
#> 6        32.3        38.3
#> 7        33.2        39.4
#> 8        41.6        56.0
```
:::

## `glmer` 

This engine requires the multilevelmod extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(multilevelmod)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_spec <- linear_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("glmer")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_fit <- 
  linear_reg_spec |> 
  fit(weight ~ Diet + Time + (1|Rat), data = reg_group_train)
#> Warning in lme4::glmer(formula = weight ~ Diet + Time + (1 | Rat), data = data,
#> : calling glmer() with family=gaussian (identity link) as a shortcut to lmer()
#> is deprecated; please call lmer() directly
linear_reg_fit
#> parsnip model object
#> 
#> Linear mixed model fit by REML ['lmerMod']
#> Formula: weight ~ Diet + Time + (1 | Rat)
#>    Data: data
#> REML criterion at convergence: 955.6549
#> Random effects:
#>  Groups   Name        Std.Dev.
#>  Rat      (Intercept) 16.331  
#>  Residual              8.117  
#> Number of obs: 132, groups:  Rat, 12
#> Fixed Effects:
#> (Intercept)        Diet2        Diet3         Time  
#>    245.4104     185.6212     259.2879       0.5492
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(linear_reg_fit, new_data = reg_group_test)
#> # A tibble: 44 × 1
#>    .pred
#>    <dbl>
#>  1  246.
#>  2  250.
#>  3  254.
#>  4  257.
#>  5  261.
#>  6  265.
#>  7  269.
#>  8  270.
#>  9  273.
#> 10  277.
#> # ℹ 34 more rows
```
:::

## `glmnet` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_spec <- linear_reg(penalty = 0.01) |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("glmnet")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_fit <- linear_reg_spec |>
  fit(strength ~ ., data = reg_train)
linear_reg_fit
#> parsnip model object
#> 
#> 
#> Call:  glmnet::glmnet(x = maybe_matrix(x), y = y, family = "gaussian") 
#> 
#>    Df  %Dev Lambda
#> 1   0  0.00 9.5680
#> 2   1  5.38 8.7180
#> 3   1  9.85 7.9430
#> 4   1 13.56 7.2380
#> 5   1 16.64 6.5950
#> 6   2 19.99 6.0090
#> 7   2 23.68 5.4750
#> 8   2 26.75 4.9890
#> 9   2 29.29 4.5450
#> 10  2 31.40 4.1420
#> 11  2 33.15 3.7740
#> 12  2 34.61 3.4380
#> 13  2 35.82 3.1330
#> 14  2 36.82 2.8550
#> 15  2 37.65 2.6010
#> 16  2 38.34 2.3700
#> 17  2 38.92 2.1590
#> 18  2 39.39 1.9680
#> 19  2 39.79 1.7930
#> 20  2 40.12 1.6340
#> 21  2 40.39 1.4880
#> 22  2 40.62 1.3560
#> 23  2 40.80 1.2360
#> 24  2 40.96 1.1260
#> 25  2 41.09 1.0260
#> 26  2 41.20 0.9348
#> 27  2 41.29 0.8517
#> 28  2 41.36 0.7761
#> 29  2 41.42 0.7071
#> 30  2 41.47 0.6443
#> 31  2 41.52 0.5871
#> 32  2 41.55 0.5349
#> 33  2 41.58 0.4874
#> 34  2 41.60 0.4441
#> 35  2 41.63 0.4046
#> 36  2 41.64 0.3687
#> 37  2 41.66 0.3359
#> 38  2 41.67 0.3061
#> 39  2 41.68 0.2789
#> 40  2 41.68 0.2541
#> 41  2 41.69 0.2316
#> 42  2 41.70 0.2110
#> 43  2 41.70 0.1922
#> 44  2 41.71 0.1752
#> 45  2 41.71 0.1596
#> 46  2 41.71 0.1454
#> 47  2 41.71 0.1325
#> 48  2 41.71 0.1207
#> 49  2 41.72 0.1100
#> 50  2 41.72 0.1002
#> 51  2 41.72 0.0913
#> 52  2 41.72 0.0832
#> 53  2 41.72 0.0758
#> 54  2 41.72 0.0691
#> 55  2 41.72 0.0630
#> 56  2 41.72 0.0574
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(linear_reg_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  32.2
#> 2  30.3
#> 3  21.7
#> 4  51.3
#> 5  40.3
#> 6  35.3
#> 7  36.3
#> 8  48.7
```
:::

## `gls` 

This engine requires the multilevelmod extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(multilevelmod)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_spec <- linear_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  # Also, nlme::gls() specifies the random effects outside of the formula so
  # we set that as an engine parameter
  set_engine("gls", correlation = nlme::corCompSymm(form = ~Time|Rat))
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_fit <- linear_reg_spec |>
  fit(weight ~ Time + Diet, data = reg_group_train)
linear_reg_fit
#> parsnip model object
#> 
#> Generalized least squares fit by REML
#>   Model: weight ~ Time + Diet 
#>   Data: data 
#>   Log-restricted-likelihood: -477.8274
#> 
#> Coefficients:
#> (Intercept)        Time       Diet2       Diet3 
#>  245.410439    0.549192  185.621212  259.287879 
#> 
#> Correlation Structure: Compound symmetry
#>  Formula: ~Time | Rat 
#>  Parameter estimate(s):
#>       Rho 
#> 0.8019221 
#> Degrees of freedom: 132 total; 128 residual
#> Residual standard error: 18.23695
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(linear_reg_fit, new_data = reg_group_test)
#> # A tibble: 44 × 1
#>    .pred
#>    <dbl>
#>  1  246.
#>  2  250.
#>  3  254.
#>  4  257.
#>  5  261.
#>  6  265.
#>  7  269.
#>  8  270.
#>  9  273.
#> 10  277.
#> # ℹ 34 more rows
```
:::

## `h2o` 

This engine requires the agua extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(agua)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_spec <- linear_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("h2o")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_fit <- linear_reg_spec |>
  fit(strength ~ ., data = reg_train)
linear_reg_fit
#> parsnip model object
#> 
#> Model Details:
#> ==============
#> 
#> H2ORegressionModel: glm
#> Model ID:  GLM_model_R_1770287512312_5934 
#> GLM Model: summary
#>     family     link                               regularization
#> 1 gaussian identity Elastic Net (alpha = 0.5, lambda = 0.01903 )
#>   number_of_predictors_total number_of_active_predictors number_of_iterations
#> 1                          2                           2                    1
#>      training_frame
#> 1 object_ujvnjgioue
#> 
#> Coefficients: glm coefficients
#>       names coefficients standardized_coefficients
#> 1 Intercept    33.577283                 33.577283
#> 2    cement     8.708461                  8.708461
#> 3       age     5.422201                  5.422201
#> 
#> H2ORegressionMetrics: glm
#> ** Reported on training data. **
#> 
#> MSE:  168.2822
#> RMSE:  12.97236
#> MAE:  10.62672
#> RMSLE:  0.4645554
#> Mean Residual Deviance :  168.2822
#> R^2 :  0.4171988
#> Null Deviance :26564.74
#> Null D.o.F. :91
#> Residual Deviance :15481.96
#> Residual D.o.F. :89
#> AIC :740.6438
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(linear_reg_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  32.1
#> 2  30.3
#> 3  21.7
#> 4  51.2
#> 5  40.3
#> 6  35.3
#> 7  36.3
#> 8  48.7
```
:::

## `keras` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_spec <- linear_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("keras")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(596)
linear_reg_fit <- linear_reg_spec |>
  fit(strength ~ ., data = reg_train)
linear_reg_fit
```
:::

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_fit
#> parsnip model object
#> 
#> Model: "sequential_3"
#> ________________________________________________________________________________
#>  Layer (type)                       Output Shape                    Param #     
#> ================================================================================
#>  dense_6 (Dense)                    (None, 1)                       3           
#>  dense_7 (Dense)                    (None, 1)                       2           
#> ================================================================================
#> Total params: 5 (20.00 Byte)
#> Trainable params: 5 (20.00 Byte)
#> Non-trainable params: 0 (0.00 Byte)
#> ________________________________________________________________________________
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(linear_reg_fit, new_data = reg_test)
#> 1/1 - 0s - 40ms/epoch - 40ms/step
#> # A tibble: 8 × 1
#>       .pred
#>       <dbl>
#> 1  0.157   
#> 2 -0.000522
#> 3 -0.0670  
#> 4  0.413   
#> 5  0.290   
#> 6  0.154   
#> 7  0.169   
#> 8  0.442
```
:::

## `lme` 

This engine requires the multilevelmod extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(multilevelmod)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_spec <- linear_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.. 
  # nlme::lme() makes us set the random effects outside of the formula so we
  # add it as an engine parameter. 
  set_engine("lme", random = ~ Time | Rat)
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_fit <- linear_reg_spec |>
  fit(weight ~ Diet + Time, data = reg_group_train)
linear_reg_fit
#> parsnip model object
#> 
#> Linear mixed-effects model fit by REML
#>   Data: data 
#>   Log-restricted-likelihood: -426.5662
#>   Fixed: weight ~ Diet + Time 
#> (Intercept)       Diet2       Diet3        Time 
#>  240.483603  199.723140  264.893298    0.549192 
#> 
#> Random effects:
#>  Formula: ~Time | Rat
#>  Structure: General positive-definite, Log-Cholesky parametrization
#>             StdDev     Corr  
#> (Intercept) 25.2657397 (Intr)
#> Time         0.3411097 -0.816
#> Residual     4.5940697       
#> 
#> Number of Observations: 132
#> Number of Groups: 12
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(linear_reg_fit, new_data = reg_group_test)
#> # A tibble: 44 × 1
#>    .pred
#>    <dbl>
#>  1  241.
#>  2  245.
#>  3  249.
#>  4  253.
#>  5  256.
#>  6  260.
#>  7  264.
#>  8  265.
#>  9  268.
#> 10  272.
#> # ℹ 34 more rows
```
:::

## `lmer` 

This engine requires the multilevelmod extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(multilevelmod)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_spec <- linear_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("lmer")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_fit <- 
  linear_reg_spec |> 
  fit(weight ~ Diet + Time + (1|Rat), data = reg_group_train)
linear_reg_fit
#> parsnip model object
#> 
#> Linear mixed model fit by REML ['lmerMod']
#> Formula: weight ~ Diet + Time + (1 | Rat)
#>    Data: data
#> REML criterion at convergence: 955.6549
#> Random effects:
#>  Groups   Name        Std.Dev.
#>  Rat      (Intercept) 16.331  
#>  Residual              8.117  
#> Number of obs: 132, groups:  Rat, 12
#> Fixed Effects:
#> (Intercept)        Diet2        Diet3         Time  
#>    245.4104     185.6212     259.2879       0.5492
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(linear_reg_fit, new_data = reg_group_test)
#> # A tibble: 44 × 1
#>    .pred
#>    <dbl>
#>  1  246.
#>  2  250.
#>  3  254.
#>  4  257.
#>  5  261.
#>  6  265.
#>  7  269.
#>  8  270.
#>  9  273.
#> 10  277.
#> # ℹ 34 more rows
```
:::

## `stan` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_spec <- linear_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("stan")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(357)
linear_reg_fit <- linear_reg_spec |>
  fit(weight ~ Diet + Time, data = reg_group_train)
linear_reg_fit
#> parsnip model object
#> 
#> stan_glm
#>  family:       gaussian [identity]
#>  formula:      weight ~ Diet + Time
#>  observations: 132
#>  predictors:   4
#> ------
#>             Median MAD_SD
#> (Intercept) 245.3    3.3 
#> Diet2       185.6    3.6 
#> Diet3       259.3    3.4 
#> Time          0.6    0.1 
#> 
#> Auxiliary parameter(s):
#>       Median MAD_SD
#> sigma 16.6    1.0  
#> 
#> ------
#> * For help interpreting the printed output see ?print.stanreg
#> * For info on the priors used see ?prior_summary.stanreg
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(linear_reg_fit, new_data = reg_group_test)
#> # A tibble: 44 × 1
#>    .pred
#>    <dbl>
#>  1  246.
#>  2  250.
#>  3  254.
#>  4  257.
#>  5  261.
#>  6  265.
#>  7  269.
#>  8  270.
#>  9  273.
#> 10  277.
#> # ℹ 34 more rows
predict(linear_reg_fit, type = "conf_int", new_data = reg_group_test)
#> # A tibble: 44 × 2
#>    .pred_lower .pred_upper
#>          <dbl>       <dbl>
#>  1        240.        252.
#>  2        244.        255.
#>  3        249.        258.
#>  4        253.        262.
#>  5        257.        265.
#>  6        261.        269.
#>  7        265.        273.
#>  8        265.        274.
#>  9        268.        278.
#> 10        271.        282.
#> # ℹ 34 more rows
predict(linear_reg_fit, type = "pred_int", new_data = reg_group_test)
#> # A tibble: 44 × 2
#>    .pred_lower .pred_upper
#>          <dbl>       <dbl>
#>  1        213.        278.
#>  2        216.        282.
#>  3        220.        287.
#>  4        224.        290.
#>  5        228.        292.
#>  6        230.        297.
#>  7        236.        301.
#>  8        236.        302.
#>  9        240.        305.
#> 10        244.        310.
#> # ℹ 34 more rows
```
:::

## `stan_glmer` 

This engine requires the multilevelmod extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(multilevelmod)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_spec <- linear_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("stan_glmer")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(895)
linear_reg_fit <- 
  linear_reg_spec |> 
  fit(weight ~ Diet + Time + (1|Rat), data = reg_group_train)
linear_reg_fit
#> parsnip model object
#> 
#> stan_glmer
#>  family:       gaussian [identity]
#>  formula:      weight ~ Diet + Time + (1 | Rat)
#>  observations: 132
#> ------
#>             Median MAD_SD
#> (Intercept) 245.6    6.8 
#> Diet2       185.7   11.5 
#> Diet3       259.2   11.5 
#> Time          0.5    0.0 
#> 
#> Auxiliary parameter(s):
#>       Median MAD_SD
#> sigma 8.2    0.5   
#> 
#> Error terms:
#>  Groups   Name        Std.Dev.
#>  Rat      (Intercept) 17.2    
#>  Residual              8.2    
#> Num. levels: Rat 12 
#> 
#> ------
#> * For help interpreting the printed output see ?print.stanreg
#> * For info on the priors used see ?prior_summary.stanreg
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(linear_reg_fit, new_data = reg_group_test)
#> # A tibble: 44 × 1
#>    .pred
#>    <dbl>
#>  1  246.
#>  2  250.
#>  3  254.
#>  4  258.
#>  5  262.
#>  6  266.
#>  7  269.
#>  8  270.
#>  9  273.
#> 10  277.
#> # ℹ 34 more rows
predict(linear_reg_fit, type = "pred_int", new_data = reg_group_test)
#> # A tibble: 44 × 2
#>    .pred_lower .pred_upper
#>          <dbl>       <dbl>
#>  1        205.        285.
#>  2        211.        289.
#>  3        214.        292.
#>  4        218.        295.
#>  5        221.        300.
#>  6        225.        303.
#>  7        230.        307.
#>  8        230.        309.
#>  9        233.        312.
#> 10        237.        314.
#> # ℹ 34 more rows
```
:::

## `spark` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_spec <- linear_reg() |> 
  set_engine("spark")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_fit <- linear_reg_spec |>
  fit(compressive_strength ~ ., data = tbl_reg$training)
linear_reg_fit
#> parsnip model object
#> 
#> Formula: compressive_strength ~ .
#> 
#> Coefficients:
#>        (Intercept)             cement blast_furnace_slag            fly_ash 
#>       -21.80239627         0.12003251         0.10399582         0.08747677 
#>              water   superplasticizer   coarse_aggregate     fine_aggregate 
#>        -0.15701342         0.28531613         0.01777782         0.02018358 
#>                age 
#>         0.11678247
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(linear_reg_fit, new_data = tbl_reg$test)
#> # Source:   SQL [?? x 1]
#> # Database: spark_connection
#>     pred
#>    <dbl>
#>  1  16.5
#>  2  19.7
#>  3  26.1
#>  4  23.6
#>  5  24.2
#>  6  29.1
#>  7  21.3
#>  8  24.2
#>  9  33.9
#> 10  57.7
#> # ℹ more rows
```
:::

:::

### Multivariate Adaptive Regression Splines (`mars()`)

:::{.panel-tabset}

## `earth`

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
mars_spec <- mars() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because earth is the default.
  set_mode("regression")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
mars_fit <- mars_spec |>
  fit(strength ~ ., data = reg_train)
mars_fit
#> parsnip model object
#> 
#> Selected 4 of 9 terms, and 2 of 2 predictors
#> Termination condition: RSq changed by less than 0.001 at 9 terms
#> Importance: age, cement
#> Number of terms at each degree of interaction: 1 3 (additive model)
#> GCV 113.532    RSS 8915.965    GRSq 0.6153128    RSq 0.6643684
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(mars_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  22.0
#> 2  43.1
#> 3  28.1
#> 4  58.0
#> 5  33.8
#> 6  34.9
#> 7  36.3
#> 8  43.5
```
:::

:::

### Neural Networks (`mlp()`)

:::{.panel-tabset}

## `nnet`

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
mlp_spec <- mlp() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because nnet is the default.
  set_mode("regression")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(159)
mlp_fit <- mlp_spec |>
  fit(strength ~ ., data = reg_train)
mlp_fit
#> parsnip model object
#> 
#> a 2-5-1 network with 21 weights
#> inputs: cement age 
#> output(s): strength 
#> options were - linear output units
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(mlp_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  14.8
#> 2  38.5
#> 3  32.0
#> 4  63.6
#> 5  43.5
#> 6  42.7
#> 7  42.3
#> 8  33.1
```
:::

## `brulee` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
mlp_spec <- mlp() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("regression") |>
  set_engine("brulee")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(407)
mlp_fit <- mlp_spec |>
  fit(strength ~ ., data = reg_train)
mlp_fit
#> parsnip model object
#> 
#> Multilayer perceptron
#> 
#> relu activation,
#> 3 hidden units,
#> 13 model parameters
#> 92 samples, 2 features, numeric outcome 
#> weight decay: 0.001 
#> dropout proportion: 0 
#> batch size: 83 
#> learn rate: 0.01 
#> scaled validation loss after 9 epochs: 0.189
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(mlp_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  23.1
#> 2  39.4
#> 3  26.9
#> 4  56.4
#> 5  32.9
#> 6  37.2
#> 7  38.4
#> 8  40.1
```
:::

## `brulee_two_layer` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
mlp_spec <- mlp() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("regression") |>
  set_engine("brulee_two_layer")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(585)
mlp_fit <- mlp_spec |>
  fit(strength ~ ., data = reg_train)
mlp_fit
#> parsnip model object
#> 
#> Multilayer perceptron
#> 
#> c(relu,relu) activation,
#> c(3,3) hidden units,
#> 25 model parameters
#> 92 samples, 2 features, numeric outcome 
#> weight decay: 0.001 
#> dropout proportion: 0 
#> batch size: 83 
#> learn rate: 0.01 
#> scaled validation loss after 3 epochs: 0.379
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(mlp_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  23.5
#> 2  32.6
#> 3  24.6
#> 4  50.5
#> 5  46.7
#> 6  33.8
#> 7  37.0
#> 8  50.5
```
:::

## `h2o` 

This engine requires the agua extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(agua)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
mlp_spec <- mlp() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("regression") |>
  set_engine("h2o")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(93)
mlp_fit <- mlp_spec |>
  fit(strength ~ ., data = reg_train)
mlp_fit
#> parsnip model object
#> 
#> Model Details:
#> ==============
#> 
#> H2ORegressionModel: deeplearning
#> Model ID:  DeepLearning_model_R_1770287512312_5935 
#> Status of Neuron Layers: predicting .outcome, regression, gaussian distribution, Quadratic loss, 801 weights/biases, 14.5 KB, 920 training samples, mini-batch size 1
#>   layer units      type dropout       l1       l2 mean_rate rate_rms momentum
#> 1     1     2     Input  0.00 %       NA       NA        NA       NA       NA
#> 2     2   200 Rectifier  0.00 % 0.000000 0.000000  0.009004 0.020737 0.000000
#> 3     3     1    Linear      NA 0.000000 0.000000  0.000818 0.000240 0.000000
#>   mean_weight weight_rms mean_bias bias_rms
#> 1          NA         NA        NA       NA
#> 2   -0.014857   0.124650  0.494931 0.011050
#> 3   -0.000526   0.099345  0.006964 0.000000
#> 
#> 
#> H2ORegressionMetrics: deeplearning
#> ** Reported on training data. **
#> ** Metrics reported on full training frame **
#> 
#> MSE:  136.9811
#> RMSE:  11.70389
#> MAE:  9.370161
#> RMSLE:  0.4189636
#> Mean Residual Deviance :  136.9811
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(mlp_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  30.2
#> 2  32.5
#> 3  21.2
#> 4  51.8
#> 5  38.6
#> 6  35.3
#> 7  36.3
#> 8  46.8
```
:::

## `keras` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
mlp_spec <- mlp() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("regression") |>
  set_engine("keras")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(879)
mlp_fit <- mlp_spec |>
  fit(strength ~ ., data = reg_train)
```
:::

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_fit
#> parsnip model object
#> 
#> Formula: compressive_strength ~ .
#> 
#> Coefficients:
#>        (Intercept)             cement blast_furnace_slag            fly_ash 
#>       -21.80239627         0.12003251         0.10399582         0.08747677 
#>              water   superplasticizer   coarse_aggregate     fine_aggregate 
#>        -0.15701342         0.28531613         0.01777782         0.02018358 
#>                age 
#>         0.11678247
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(mlp_fit, new_data = reg_test)
#> 1/1 - 0s - 39ms/epoch - 39ms/step
#> # A tibble: 8 × 1
#>    .pred
#>    <dbl>
#> 1 -0.386
#> 2 -0.337
#> 3 -0.299
#> 4 -0.278
#> 5 -0.384
#> 6 -0.373
#> 7 -0.373
#> 8 -0.341
```
:::

:::

### K-Nearest Neighbors (`nearest_neighbor()`)

:::{.panel-tabset}

## `kknn`

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
nearest_neighbor_spec <- nearest_neighbor() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because kknn is the default.
  set_mode("regression")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
nearest_neighbor_fit <- nearest_neighbor_spec |>
  fit(strength ~ ., data = reg_train)
nearest_neighbor_fit
#> parsnip model object
#> 
#> 
#> Call:
#> kknn::train.kknn(formula = strength ~ ., data = data, ks = min_rows(5,     data, 5))
#> 
#> Type of response variable: continuous
#> minimal mean absolute error: 8.257735
#> Minimal mean squared error: 115.8737
#> Best kernel: optimal
#> Best k: 5
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(nearest_neighbor_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  16.3
#> 2  35.7
#> 3  27.5
#> 4  56.7
#> 5  42.6
#> 6  41.7
#> 7  41.2
#> 8  50.2
```
:::

### Null Model (`null_model()`)

## `parsnip`

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
null_model_spec <- null_model() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because parsnip is the default.
  set_mode("regression")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
null_model_fit <- null_model_spec |>
  fit(strength ~ ., data = reg_train)
null_model_fit
#> parsnip model object
#> 
#> Null Classification Model
#> Predicted Value: 33.57728
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(null_model_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  33.6
#> 2  33.6
#> 3  33.6
#> 4  33.6
#> 5  33.6
#> 6  33.6
#> 7  33.6
#> 8  33.6
```
:::

:::

### Partial Least Squares (`pls()`)

:::{.panel-tabset}

## `mixOmics`

This engine requires the plsmod extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(plsmod)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
pls_spec <- pls() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because mixOmics is the default.
  set_mode("regression")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
pls_fit <- pls_spec |>
  fit(strength ~ ., data = reg_train)
pls_fit
#> parsnip model object
#> 
#> 
#> Call:
#>  mixOmics::spls(X = x, Y = y, ncomp = ncomp, keepX = keepX) 
#> 
#>  sPLS with a 'regression' mode with 2 sPLS components. 
#>  You entered data X of dimensions: 92 2 
#>  You entered data Y of dimensions: 92 1 
#> 
#>  Selection of [2] [2] variables on each of the sPLS components on the X data set. 
#>  Selection of [1] [1] variables on each of the sPLS components on the Y data set. 
#> 
#>  Main numerical outputs: 
#>  -------------------- 
#>  loading vectors: see object$loadings 
#>  variates: see object$variates 
#>  variable names: see object$names 
#> 
#>  Functions to visualise samples: 
#>  -------------------- 
#>  plotIndiv, plotArrow 
#> 
#>  Functions to visualise variables: 
#>  -------------------- 
#>  plotVar, plotLoadings, network, cim
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(pls_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  32.1
#> 2  30.3
#> 3  21.6
#> 4  51.4
#> 5  40.3
#> 6  35.3
#> 7  36.3
#> 8  48.8
```
:::

:::

### Poisson Reg (`poisson_reg()`)

:::{.panel-tabset}

## `glm` 

This engine requires the poissonreg extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(poissonreg)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
# This model only works with a single mode, so we don't need to set the mode.
# We don't need to set the engine because glm is the default.
poisson_reg_spec <- poisson_reg()
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
poisson_reg_fit <- poisson_reg_spec |>
  fit(num_years ~ ., data = count_train)
poisson_reg_fit
#> parsnip model object
#> 
#> 
#> Call:  stats::glm(formula = num_years ~ ., family = stats::poisson, 
#>     data = data)
#> 
#> Coefficients:
#> (Intercept)          age       income  
#>      2.2861       0.2804       0.2822  
#> 
#> Degrees of Freedom: 1460 Total (i.e. Null);  1458 Residual
#> Null Deviance:	    7434 
#> Residual Deviance: 2597 	AIC: 8446
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(poisson_reg_fit, new_data = count_test)
#> # A tibble: 9 × 1
#>   .pred
#>   <dbl>
#> 1 31.6 
#> 2  6.66
#> 3 11.8 
#> 4 24.8 
#> 5 26.6 
#> 6  8.23
#> 7 32.1 
#> 8  4.86
#> 9 28.3
```
:::

## `gee` 

This engine requires the multilevelmod extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(multilevelmod)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
poisson_reg_spec <- poisson_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("gee")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
poisson_reg_fit <- 
  poisson_reg_spec |> 
  fit(weight ~ Diet + Time + id_var(Rat), data = reg_group_train)
#> Beginning Cgee S-function, @(#) geeformula.q 4.13 98/01/27
#> running glm to get initial regression estimate
poisson_reg_fit
#> parsnip model object
#> 
#> 
#>  GEE:  GENERALIZED LINEAR MODELS FOR DEPENDENT DATA
#>  gee S-function, version 4.13 modified 98/01/27 (1998) 
#> 
#> Model:
#>  Link:                      Logarithm 
#>  Variance to Mean Relation: Poisson 
#>  Correlation Structure:     Independent 
#> 
#> Call:
#> gee::gee(formula = weight ~ Diet + Time, id = data$Rat, data = data, 
#>     family = stats::poisson)
#> 
#> Number of observations :  132 
#> 
#> Maximum cluster size   :  11 
#> 
#> 
#> Coefficients:
#> (Intercept)       Diet2       Diet3        Time 
#> 5.525683187 0.532717136 0.684495610 0.001467487 
#> 
#> Estimated Scale Parameter:  0.6879328
#> Number of Iterations:  1
#> 
#> Working Correlation[1:4,1:4]
#>      [,1] [,2] [,3] [,4]
#> [1,]    1    0    0    0
#> [2,]    0    1    0    0
#> [3,]    0    0    1    0
#> [4,]    0    0    0    1
#> 
#> 
#> Returned Error Value:
#> [1] 0
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Can't reproduce this:
# predict(poisson_reg_fit, new_data = reg_group_test)
```
:::

## `glmer` 

This engine requires the multilevelmod extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(multilevelmod)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
poisson_reg_spec <- poisson_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("glmer")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(826)
poisson_reg_fit <- 
  poisson_reg_spec |> 
  fit(weight ~ Diet + Time + (1|Rat), data = reg_group_train)
#> Warning in checkConv(attr(opt, "derivs"), opt$par, ctrl = control$checkConv, : Model failed to converge with max|grad| = 0.00394285 (tol = 0.002, component 1)
#>   See ?lme4::convergence and ?lme4::troubleshooting.
#> Warning in checkConv(attr(opt, "derivs"), opt$par, ctrl = control$checkConv, : Model is nearly unidentifiable: very large eigenvalue
#>  - Rescale variables?
poisson_reg_fit
#> parsnip model object
#> 
#> Generalized linear mixed model fit by maximum likelihood (Laplace
#>   Approximation) [glmerMod]
#>  Family: poisson  ( log )
#> Formula: weight ~ Diet + Time + (1 | Rat)
#>    Data: data
#>       AIC       BIC    logLik -2*log(L)  df.resid 
#> 1079.1349 1093.5489 -534.5675 1069.1349       127 
#> Random effects:
#>  Groups Name        Std.Dev.
#>  Rat    (Intercept) 0.03683 
#> Number of obs: 132, groups:  Rat, 12
#> Fixed Effects:
#> (Intercept)        Diet2        Diet3         Time  
#>    5.524796     0.533446     0.684637     0.001467  
#> optimizer (Nelder_Mead) convergence code: 0 (OK) ; 0 optimizer warnings; 2 lme4 warnings
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(poisson_reg_fit, new_data = reg_group_test)
#> # A tibble: 44 × 1
#>    .pred
#>    <dbl>
#>  1  251.
#>  2  254.
#>  3  256.
#>  4  259.
#>  5  262.
#>  6  264.
#>  7  267.
#>  8  268.
#>  9  270.
#> 10  273.
#> # ℹ 34 more rows
```
:::

## `glmnet` 

This engine requires the poissonreg extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(poissonreg)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
poisson_reg_spec <- poisson_reg(penalty = 0.01) |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("glmnet")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
poisson_reg_fit <- poisson_reg_spec |>
  fit(num_years ~ ., data = count_train)
poisson_reg_fit
#> parsnip model object
#> 
#> 
#> Call:  glmnet::glmnet(x = maybe_matrix(x), y = y, family = "poisson") 
#> 
#>    Df  %Dev Lambda
#> 1   0  0.00 5.9710
#> 2   1 10.26 5.4400
#> 3   1 18.31 4.9570
#> 4   2 24.84 4.5170
#> 5   2 32.06 4.1150
#> 6   2 37.94 3.7500
#> 7   2 42.73 3.4170
#> 8   2 46.65 3.1130
#> 9   2 49.87 2.8370
#> 10  2 52.51 2.5850
#> 11  2 54.69 2.3550
#> 12  2 56.48 2.1460
#> 13  2 57.96 1.9550
#> 14  2 59.18 1.7810
#> 15  2 60.19 1.6230
#> 16  2 61.03 1.4790
#> 17  2 61.72 1.3480
#> 18  2 62.29 1.2280
#> 19  2 62.76 1.1190
#> 20  2 63.16 1.0190
#> 21  2 63.48 0.9289
#> 22  2 63.75 0.8463
#> 23  2 63.98 0.7712
#> 24  2 64.16 0.7026
#> 25  2 64.31 0.6402
#> 26  2 64.44 0.5833
#> 27  2 64.55 0.5315
#> 28  2 64.64 0.4843
#> 29  2 64.71 0.4413
#> 30  2 64.77 0.4021
#> 31  2 64.82 0.3664
#> 32  2 64.86 0.3338
#> 33  2 64.90 0.3042
#> 34  2 64.92 0.2771
#> 35  2 64.95 0.2525
#> 36  2 64.97 0.2301
#> 37  2 64.98 0.2096
#> 38  2 65.00 0.1910
#> 39  2 65.01 0.1741
#> 40  2 65.02 0.1586
#> 41  2 65.03 0.1445
#> 42  2 65.03 0.1317
#> 43  2 65.04 0.1200
#> 44  2 65.04 0.1093
#> 45  2 65.05 0.0996
#> 46  2 65.05 0.0907
#> 47  2 65.05 0.0827
#> 48  2 65.05 0.0753
#> 49  2 65.06 0.0687
#> 50  2 65.06 0.0625
#> 51  2 65.06 0.0570
#> 52  2 65.06 0.0519
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(poisson_reg_fit, new_data = count_test)
#> # A tibble: 9 × 1
#>   .pred
#>   <dbl>
#> 1 31.4 
#> 2  6.70
#> 3 11.8 
#> 4 24.6 
#> 5 26.4 
#> 6  8.27
#> 7 31.8 
#> 8  4.91
#> 9 28.1
```
:::

## `h2o` 

This engine requires the agua extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(agua)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
poisson_reg_spec <- poisson_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("h2o")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
poisson_reg_fit <- poisson_reg_spec |>
  fit(num_years ~ ., data = count_train)
poisson_reg_fit
#> parsnip model object
#> 
#> Model Details:
#> ==============
#> 
#> H2ORegressionModel: glm
#> Model ID:  GLM_model_R_1770287512312_5936 
#> GLM Model: summary
#>    family link                               regularization
#> 1 poisson  log Elastic Net (alpha = 0.5, lambda = 0.01194 )
#>   number_of_predictors_total number_of_active_predictors number_of_iterations
#> 1                          2                           2                    4
#>      training_frame
#> 1 object_kyirzmfbti
#> 
#> Coefficients: glm coefficients
#>       names coefficients standardized_coefficients
#> 1 Intercept     2.286411                  2.286411
#> 2       age     0.279967                  0.279967
#> 3    income     0.281952                  0.281952
#> 
#> H2ORegressionMetrics: glm
#> ** Reported on training data. **
#> 
#> MSE:  18.40519
#> RMSE:  4.290128
#> MAE:  3.297048
#> RMSLE:  0.467537
#> Mean Residual Deviance :  1.777749
#> R^2 :  0.6934292
#> Null Deviance :7434.374
#> Null D.o.F. :1460
#> Residual Deviance :2597.291
#> Residual D.o.F. :1458
#> AIC :8445.967
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(poisson_reg_fit, new_data = count_test)
#> # A tibble: 9 × 1
#>   .pred
#>   <dbl>
#> 1 31.6 
#> 2  6.67
#> 3 11.8 
#> 4 24.8 
#> 5 26.5 
#> 6  8.24
#> 7 32.0 
#> 8  4.87
#> 9 28.2
```
:::

## `hurdle` 

This engine requires the poissonreg extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(poissonreg)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
poisson_reg_spec <- poisson_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("hurdle")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
poisson_reg_fit <- poisson_reg_spec |>
  fit(num_years ~ ., data = count_train)
poisson_reg_fit
#> parsnip model object
#> 
#> 
#> Call:
#> pscl::hurdle(formula = num_years ~ ., data = data)
#> 
#> Count model coefficients (truncated poisson with log link):
#> (Intercept)          age       income  
#>      2.2911       0.2749       0.2820  
#> 
#> Zero hurdle model coefficients (binomial with logit link):
#> (Intercept)          age       income  
#>      24.656        5.611       13.092
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(poisson_reg_fit, new_data = count_test)
#> # A tibble: 9 × 1
#>   .pred
#>   <dbl>
#> 1 31.5 
#> 2  6.74
#> 3 11.9 
#> 4 24.6 
#> 5 26.4 
#> 6  8.32
#> 7 31.9 
#> 8  4.89
#> 9 28.2
```
:::

## `stan` 

This engine requires the poissonreg extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(poissonreg)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
poisson_reg_spec <- poisson_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("stan")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(213)
poisson_reg_fit <- 
  poisson_reg_spec |> 
  fit(weight ~ Diet + Time, data = reg_group_train)
#> 
#> SAMPLING FOR MODEL 'count' NOW (CHAIN 1).
#> Chain 1: 
#> Chain 1: Gradient evaluation took 0.000103 seconds
#> Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 1.03 seconds.
#> Chain 1: Adjust your expectations accordingly!
#> Chain 1: 
#> Chain 1: 
#> Chain 1: Iteration:    1 / 2000 [  0%]  (Warmup)
#> Chain 1: Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Chain 1: Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Chain 1: Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Chain 1: Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Chain 1: Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Chain 1: Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Chain 1: Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Chain 1: Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Chain 1: Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Chain 1: Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Chain 1: Iteration: 2000 / 2000 [100%]  (Sampling)
#> Chain 1: 
#> Chain 1:  Elapsed Time: 0.026 seconds (Warm-up)
#> Chain 1:                0.026 seconds (Sampling)
#> Chain 1:                0.052 seconds (Total)
#> Chain 1: 
#> 
#> SAMPLING FOR MODEL 'count' NOW (CHAIN 2).
#> Chain 2: 
#> Chain 2: Gradient evaluation took 5e-06 seconds
#> Chain 2: 1000 transitions using 10 leapfrog steps per transition would take 0.05 seconds.
#> Chain 2: Adjust your expectations accordingly!
#> Chain 2: 
#> Chain 2: 
#> Chain 2: Iteration:    1 / 2000 [  0%]  (Warmup)
#> Chain 2: Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Chain 2: Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Chain 2: Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Chain 2: Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Chain 2: Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Chain 2: Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Chain 2: Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Chain 2: Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Chain 2: Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Chain 2: Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Chain 2: Iteration: 2000 / 2000 [100%]  (Sampling)
#> Chain 2: 
#> Chain 2:  Elapsed Time: 0.027 seconds (Warm-up)
#> Chain 2:                0.027 seconds (Sampling)
#> Chain 2:                0.054 seconds (Total)
#> Chain 2: 
#> 
#> SAMPLING FOR MODEL 'count' NOW (CHAIN 3).
#> Chain 3: 
#> Chain 3: Gradient evaluation took 4e-06 seconds
#> Chain 3: 1000 transitions using 10 leapfrog steps per transition would take 0.04 seconds.
#> Chain 3: Adjust your expectations accordingly!
#> Chain 3: 
#> Chain 3: 
#> Chain 3: Iteration:    1 / 2000 [  0%]  (Warmup)
#> Chain 3: Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Chain 3: Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Chain 3: Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Chain 3: Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Chain 3: Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Chain 3: Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Chain 3: Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Chain 3: Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Chain 3: Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Chain 3: Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Chain 3: Iteration: 2000 / 2000 [100%]  (Sampling)
#> Chain 3: 
#> Chain 3:  Elapsed Time: 0.026 seconds (Warm-up)
#> Chain 3:                0.027 seconds (Sampling)
#> Chain 3:                0.053 seconds (Total)
#> Chain 3: 
#> 
#> SAMPLING FOR MODEL 'count' NOW (CHAIN 4).
#> Chain 4: 
#> Chain 4: Gradient evaluation took 4e-06 seconds
#> Chain 4: 1000 transitions using 10 leapfrog steps per transition would take 0.04 seconds.
#> Chain 4: Adjust your expectations accordingly!
#> Chain 4: 
#> Chain 4: 
#> Chain 4: Iteration:    1 / 2000 [  0%]  (Warmup)
#> Chain 4: Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Chain 4: Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Chain 4: Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Chain 4: Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Chain 4: Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Chain 4: Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Chain 4: Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Chain 4: Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Chain 4: Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Chain 4: Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Chain 4: Iteration: 2000 / 2000 [100%]  (Sampling)
#> Chain 4: 
#> Chain 4:  Elapsed Time: 0.027 seconds (Warm-up)
#> Chain 4:                0.031 seconds (Sampling)
#> Chain 4:                0.058 seconds (Total)
#> Chain 4:
poisson_reg_fit
#> parsnip model object
#> 
#> stan_glm
#>  family:       poisson [log]
#>  formula:      weight ~ Diet + Time
#>  observations: 132
#>  predictors:   4
#> ------
#>             Median MAD_SD
#> (Intercept) 5.5    0.0   
#> Diet2       0.5    0.0   
#> Diet3       0.7    0.0   
#> Time        0.0    0.0   
#> 
#> ------
#> * For help interpreting the printed output see ?print.stanreg
#> * For info on the priors used see ?prior_summary.stanreg
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(poisson_reg_fit, new_data = reg_group_test)
#> # A tibble: 44 × 1
#>    .pred
#>    <dbl>
#>  1  5.53
#>  2  5.54
#>  3  5.55
#>  4  5.56
#>  5  5.57
#>  6  5.58
#>  7  5.59
#>  8  5.59
#>  9  5.60
#> 10  5.61
#> # ℹ 34 more rows
predict(poisson_reg_fit, type = "conf_int", new_data = reg_group_test)
#> Instead of posterior_linpred(..., transform=TRUE) please call posterior_epred(), which provides equivalent functionality.
#> # A tibble: 44 × 2
#>    .pred_lower .pred_upper
#>          <dbl>       <dbl>
#>  1        246.        257.
#>  2        249.        259.
#>  3        252.        261.
#>  4        255.        263.
#>  5        258.        266.
#>  6        261.        269.
#>  7        263.        272.
#>  8        264.        272.
#>  9        266.        275.
#> 10        268.        278.
#> # ℹ 34 more rows
predict(poisson_reg_fit, type = "pred_int", new_data = reg_group_test)
#> # A tibble: 44 × 2
#>    .pred_lower .pred_upper
#>          <dbl>       <dbl>
#>  1         220         284
#>  2         222         286
#>  3         225         288
#>  4         228         291
#>  5         230         296
#>  6         232         297
#>  7         235         300
#>  8         236         300
#>  9         238         303
#> 10         241         306
#> # ℹ 34 more rows
```
:::

## `stan_glmer` 

This engine requires the multilevelmod extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(multilevelmod)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
poisson_reg_spec <- poisson_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("stan_glmer")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(690)
poisson_reg_fit <- 
  poisson_reg_spec |> 
  fit(weight ~ Diet + Time + (1|Rat), data = reg_group_train)
poisson_reg_fit
#> parsnip model object
#> 
#> stan_glmer
#>  family:       poisson [log]
#>  formula:      weight ~ Diet + Time + (1 | Rat)
#>  observations: 132
#> ------
#>             Median MAD_SD
#> (Intercept) 5.5    0.0   
#> Diet2       0.5    0.0   
#> Diet3       0.7    0.0   
#> Time        0.0    0.0   
#> 
#> Error terms:
#>  Groups Name        Std.Dev.
#>  Rat    (Intercept) 0.054   
#> Num. levels: Rat 12 
#> 
#> ------
#> * For help interpreting the printed output see ?print.stanreg
#> * For info on the priors used see ?prior_summary.stanreg
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(poisson_reg_fit, new_data = reg_group_test)
#> # A tibble: 44 × 1
#>    .pred
#>    <dbl>
#>  1  251.
#>  2  254.
#>  3  256.
#>  4  259.
#>  5  261.
#>  6  264.
#>  7  267.
#>  8  268.
#>  9  270.
#> 10  272.
#> # ℹ 34 more rows
predict(poisson_reg_fit, type = "pred_int", new_data = reg_group_test)
#> # A tibble: 44 × 2
#>    .pred_lower .pred_upper
#>          <dbl>       <dbl>
#>  1        210.        294 
#>  2        213         298 
#>  3        214         301 
#>  4        217         304 
#>  5        220         306 
#>  6        222         309 
#>  7        223         313.
#>  8        225         315 
#>  9        226         317.
#> 10        229         320 
#> # ℹ 34 more rows
```
:::

## `zeroinfl` 

This engine requires the poissonreg extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(poissonreg)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
poisson_reg_spec <- poisson_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("zeroinfl")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
poisson_reg_fit <- poisson_reg_spec |>
  fit(num_years ~ ., data = count_train)
#> Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
poisson_reg_fit
#> parsnip model object
#> 
#> 
#> Call:
#> pscl::zeroinfl(formula = num_years ~ ., data = data)
#> 
#> Count model coefficients (poisson with log link):
#> (Intercept)          age       income  
#>      2.2912       0.2748       0.2821  
#> 
#> Zero-inflation model coefficients (binomial with logit link):
#> (Intercept)          age       income  
#>      -48.26       -18.22       -11.72
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(poisson_reg_fit, new_data = count_test)
#> # A tibble: 9 × 1
#>   .pred
#>   <dbl>
#> 1 31.5 
#> 2  6.74
#> 3 11.9 
#> 4 24.6 
#> 5 26.4 
#> 6  8.31
#> 7 31.9 
#> 8  4.93
#> 9 28.2
```
:::

:::

### Random Forests (`rand_forest()`)

:::{.panel-tabset}

## `ranger`

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
rand_forest_spec <- rand_forest() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because ranger is the default.
  set_engine("ranger", keep.inbag = TRUE) |> 
  # However, we'll set the engine and use the keep.inbag=TRUE option so that we 
  # can produce interval predictions. This is not generally required. 
  set_mode("regression")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(860)
rand_forest_fit <- rand_forest_spec |>
  fit(strength ~ ., data = reg_train)
rand_forest_fit
#> parsnip model object
#> 
#> Ranger result
#> 
#> Call:
#>  ranger::ranger(x = maybe_data_frame(x), y = y, keep.inbag = ~TRUE,      num.threads = 1, verbose = FALSE, seed = sample.int(10^5,          1)) 
#> 
#> Type:                             Regression 
#> Number of trees:                  500 
#> Sample size:                      92 
#> Number of independent variables:  2 
#> Mtry:                             1 
#> Target node size:                 5 
#> Variable importance mode:         none 
#> Splitrule:                        variance 
#> OOB prediction error (MSE):       92.94531 
#> R squared (OOB):                  0.6816071
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(rand_forest_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  23.6
#> 2  36.9
#> 3  28.4
#> 4  56.5
#> 5  38.6
#> 6  36.5
#> 7  38.7
#> 8  34.4
predict(rand_forest_fit, type = "conf_int", new_data = reg_test)
#> Warning in rInfJack(pred = result$predictions, inbag = inbag.counts, used.trees
#> = 1:num.trees): Sample size <=20, no calibration performed.
#> # A tibble: 8 × 2
#>   .pred_lower .pred_upper
#>         <dbl>       <dbl>
#> 1        18.1        29.1
#> 2        32.6        41.1
#> 3        24.0        32.9
#> 4        45.4        67.7
#> 5        33.0        44.3
#> 6        32.0        41.0
#> 7        35.1        42.3
#> 8        28.4        40.3
```
:::

## `aorsf` 

This engine requires the bonsai extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(bonsai)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
rand_forest_spec <- rand_forest() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("regression") |>
  set_engine("aorsf")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(47)
rand_forest_fit <- rand_forest_spec |>
  fit(strength ~ ., data = reg_train)
rand_forest_fit
#> parsnip model object
#> 
#> ---------- Oblique random regression forest
#> 
#>      Linear combinations: Accelerated Linear regression
#>           N observations: 92
#>                  N trees: 500
#>       N predictors total: 2
#>    N predictors per node: 2
#>  Average leaves per tree: 13.994
#> Min observations in leaf: 5
#>           OOB stat value: 0.59
#>            OOB stat type: RSQ
#>      Variable importance: anova
#> 
#> -----------------------------------------
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(rand_forest_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  25.2
#> 2  36.4
#> 3  29.7
#> 4  55.5
#> 5  42.3
#> 6  38.5
#> 7  40.7
#> 8  52.7
```
:::

## `grf` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
rand_forest_spec <- rand_forest() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("regression") |>
  set_engine("grf")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(130)
rand_forest_fit <- rand_forest_spec |>
  fit(strength ~ ., data = reg_train)
rand_forest_fit
#> parsnip model object
#> 
#> GRF forest object of type regression_forest 
#> Number of trees: 2000 
#> Number of training samples: 92 
#> Variable importance: 
#>    1    2 
#> 0.51 0.49
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(rand_forest_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  26.8
#> 2  38.9
#> 3  28.3
#> 4  47.0
#> 5  41.1
#> 6  36.4
#> 7  38.3
#> 8  33.8
predict(rand_forest_fit, type = "conf_int", new_data = reg_test)
#> # A tibble: 8 × 2
#>   .pred_lower .pred_upper
#>         <dbl>       <dbl>
#> 1        34.8        18.8
#> 2        47.1        30.8
#> 3        31.9        24.7
#> 4        58.3        35.8
#> 5        48.0        34.1
#> 6        40.0        32.9
#> 7        43.7        32.9
#> 8        43.9        23.7
```
:::

## `h2o` 

This engine requires the agua extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(agua)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
rand_forest_spec <- rand_forest() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("regression") |>
  set_engine("h2o")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(211)
rand_forest_fit <- rand_forest_spec |>
  fit(strength ~ ., data = reg_train)
rand_forest_fit
#> parsnip model object
#> 
#> Model Details:
#> ==============
#> 
#> H2ORegressionModel: drf
#> Model ID:  DRF_model_R_1770287512312_5937 
#> Model Summary: 
#>   number_of_trees number_of_internal_trees model_size_in_bytes min_depth
#> 1              50                       50               22316         7
#>   max_depth mean_depth min_leaves max_leaves mean_leaves
#> 1        14    9.04000         14         43    30.86000
#> 
#> 
#> H2ORegressionMetrics: drf
#> ** Reported on training data. **
#> ** Metrics reported on Out-Of-Bag training samples **
#> 
#> MSE:  89.19785
#> RMSE:  9.444462
#> MAE:  7.597463
#> RMSLE:  0.3303384
#> Mean Residual Deviance :  89.19785
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(rand_forest_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  24.9
#> 2  36.4
#> 3  28.1
#> 4  56.8
#> 5  39.0
#> 6  37.8
#> 7  37.4
#> 8  31.8
```
:::

## `partykit` 

This engine requires the bonsai extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(bonsai)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
rand_forest_spec <- rand_forest() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("regression") |>
  set_engine("partykit")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(981)
rand_forest_fit <- rand_forest_spec |>
  fit(strength ~ ., data = reg_train)
```
:::

The print method has a lot of output: 

<details>

::: {.cell layout-align="center"}

```{.r .cell-code}
capture.output(print(rand_forest_fit))[1:100] |> cat(sep = "\n")
#> parsnip model object
#> 
#> $nodes
#> $nodes[[1]]
#> [1] root
#> |   [2] V2 <= 0.31678
#> |   |   [3] V3 <= -0.60316 *
#> |   |   [4] V3 > -0.60316
#> |   |   |   [5] V2 <= -0.89134 *
#> |   |   |   [6] V2 > -0.89134 *
#> |   [7] V2 > 0.31678
#> |   |   [8] V3 <= -0.60316 *
#> |   |   [9] V3 > -0.60316 *
#> 
#> $nodes[[2]]
#> [1] root
#> |   [2] V2 <= 0.62459
#> |   |   [3] V3 <= -0.60316 *
#> |   |   [4] V3 > -0.60316
#> |   |   |   [5] V2 <= -1.16452 *
#> |   |   |   [6] V2 > -1.16452
#> |   |   |   |   [7] V3 <= -0.2359 *
#> |   |   |   |   [8] V3 > -0.2359 *
#> |   [9] V2 > 0.62459 *
#> 
#> $nodes[[3]]
#> [1] root
#> |   [2] V2 <= 0.34564
#> |   |   [3] V3 <= -0.60316 *
#> |   |   [4] V3 > -0.60316
#> |   |   |   [5] V2 <= -1.19338 *
#> |   |   |   [6] V2 > -1.19338 *
#> |   [7] V2 > 0.34564
#> |   |   [8] V2 <= 1.21134 *
#> |   |   [9] V2 > 1.21134 *
#> 
#> $nodes[[4]]
#> [1] root
#> |   [2] V2 <= 0.34564
#> |   |   [3] V3 <= -0.60316 *
#> |   |   [4] V3 > -0.60316
#> |   |   |   [5] V3 <= 0.25377 *
#> |   |   |   [6] V3 > 0.25377 *
#> |   [7] V2 > 0.34564
#> |   |   [8] V3 <= -0.60316 *
#> |   |   [9] V3 > -0.60316 *
#> 
#> $nodes[[5]]
#> [1] root
#> |   [2] V2 <= 0.62459
#> |   |   [3] V3 <= -0.48074 *
#> |   |   [4] V3 > -0.48074
#> |   |   |   [5] V2 <= -1.12604 *
#> |   |   |   [6] V2 > -1.12604
#> |   |   |   |   [7] V3 <= -0.2359 *
#> |   |   |   |   [8] V3 > -0.2359 *
#> |   [9] V2 > 0.62459 *
#> 
#> $nodes[[6]]
#> [1] root
#> |   [2] V2 <= 0.72078
#> |   |   [3] V3 <= -0.60316 *
#> |   |   [4] V3 > -0.60316
#> |   |   |   [5] V2 <= -0.84517 *
#> |   |   |   [6] V2 > -0.84517 *
#> |   [7] V2 > 0.72078 *
#> 
#> $nodes[[7]]
#> [1] root
#> |   [2] V2 <= 0.72078
#> |   |   [3] V3 <= -0.60316 *
#> |   |   [4] V3 > -0.60316
#> |   |   |   [5] V3 <= -0.2359
#> |   |   |   |   [6] V2 <= 0.24945 *
#> |   |   |   |   [7] V2 > 0.24945 *
#> |   |   |   [8] V3 > -0.2359 *
#> |   [9] V2 > 0.72078 *
#> 
#> $nodes[[8]]
#> [1] root
#> |   [2] V2 <= 0.72078
#> |   |   [3] V3 <= -0.48074 *
#> |   |   [4] V3 > -0.48074
#> |   |   |   [5] V3 <= -0.2359 *
#> |   |   |   [6] V3 > -0.2359 *
#> |   [7] V2 > 0.72078 *
#> 
#> $nodes[[9]]
#> [1] root
#> |   [2] V2 <= 0.62459
#> |   |   [3] V3 <= -0.60316 *
#> |   |   [4] V3 > -0.60316
#> |   |   |   [5] V2 <= -0.23149
#> |   |   |   |   [6] V2 <= -1.09526 *
#> |   |   |   |   [7] V2 > -1.09526 *
#> |   |   |   [8] V2 > -0.23149 *
#> |   [9] V2 > 0.62459 *
#> 
#> $nodes[[10]]
#> [1] root
```
:::

</details>

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(rand_forest_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  16.3
#> 2  37.7
#> 3  28.5
#> 4  50.6
#> 5  49.2
#> 6  36.1
#> 7  38.6
#> 8  49.7
```
:::

## `randomForest` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
rand_forest_spec <- rand_forest() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("regression") |>
  set_engine("randomForest")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(793)
rand_forest_fit <- rand_forest_spec |>
  fit(strength ~ ., data = reg_train)
rand_forest_fit
#> parsnip model object
#> 
#> 
#> Call:
#>  randomForest(x = maybe_data_frame(x), y = y) 
#>                Type of random forest: regression
#>                      Number of trees: 500
#> No. of variables tried at each split: 1
#> 
#>           Mean of squared residuals: 90.38475
#>                     % Var explained: 68.7
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(rand_forest_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  23.5
#> 2  36.8
#> 3  28.6
#> 4  58.0
#> 5  38.3
#> 6  35.4
#> 7  38.1
#> 8  33.7
```
:::

## `spark` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
rand_forest_spec <- rand_forest() |>
  set_engine("spark") |> 
  set_mode("regression")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(157)
rand_forest_fit <- rand_forest_spec |>
  fit(compressive_strength ~ ., data = tbl_reg$training)
rand_forest_fit
#> parsnip model object
#> 
#> Formula: compressive_strength ~ .
#> 
#> RandomForestRegressionModel: uid=random_forest__18b9f872_1e54_42c8_9671_918541c74363, numTrees=20, numFeatures=8
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(rand_forest_fit, new_data = tbl_reg$test)
#> # Source:   SQL [?? x 1]
#> # Database: spark_connection
#>     pred
#>    <dbl>
#>  1  28.2
#>  2  29.6
#>  3  23.0
#>  4  28.2
#>  5  15.2
#>  6  35.3
#>  7  18.6
#>  8  31.9
#>  9  36.3
#> 10  45.4
#> # ℹ more rows
```
:::

:::

### Rule Fit (`rule_fit()`)

:::{.panel-tabset}

## `xrf`

This engine requires the rules extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(rules)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
rule_fit_spec <- rule_fit() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because xrf is the default.
  set_mode("regression")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(431)
rule_fit_fit <- rule_fit_spec |>
  fit(strength ~ ., data = reg_train)
rule_fit_fit
#> parsnip model object
#> 
#> An eXtreme RuleFit model of 187 rules.
#> 
#> Original Formula:
#> 
#> strength ~ cement + age
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(rule_fit_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  32.0
#> 2  33.5
#> 3  31.9
#> 4  47.9
#> 5  48.2
#> 6  28.3
#> 7  38.9
#> 8  33.4
```
:::

## `h2o` 

This engine requires the agua extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(agua)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
rule_fit_spec <- rule_fit() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("regression") |>
  set_engine("h2o")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(236)
rule_fit_fit <- rule_fit_spec |>
  fit(strength ~ ., data = reg_train)
rule_fit_fit
#> parsnip model object
#> 
#> Model Details:
#> ==============
#> 
#> H2ORegressionModel: rulefit
#> Model ID:  RuleFit_model_R_1770287512312_5938 
#> Rulefit Model Summary: 
#>     family     link           regularization number_of_predictors_total
#> 1 gaussian identity Lasso (lambda = 0.9516 )                       1917
#>   number_of_active_predictors number_of_iterations rule_ensemble_size
#> 1                          51                    1               1915
#>   number_of_trees number_of_internal_trees min_depth max_depth mean_depth
#> 1             150                      150         0         5    4.00000
#>   min_leaves max_leaves mean_leaves
#> 1          0         28    12.76667
#> 
#> 
#> H2ORegressionMetrics: rulefit
#> ** Reported on training data. **
#> 
#> MSE:  90.45501
#> RMSE:  9.510784
#> MAE:  7.15224
#> RMSLE:  0.3531064
#> Mean Residual Deviance :  90.45501
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(rule_fit_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  26.9
#> 2  35.5
#> 3  26.9
#> 4  50.1
#> 5  42.1
#> 6  34.5
#> 7  39.3
#> 8  40.8
```
:::

:::

### Support Vector Machine (Linear Kernel) (`svm_linear()`)

:::{.panel-tabset}

## `kernlab`

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
svm_linear_spec <- svm_linear() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("regression") |>
  set_engine("kernlab")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
svm_linear_fit <- svm_linear_spec |>
  fit(strength ~ ., data = reg_train)
svm_linear_fit
#> parsnip model object
#> 
#> Support Vector Machine object of class "ksvm" 
#> 
#> SV type: eps-svr  (regression) 
#>  parameter : epsilon = 0.1  cost C = 1 
#> 
#> Linear (vanilla) kernel function. 
#> 
#> Number of Support Vectors : 85 
#> 
#> Objective Function Value : -47.4495 
#> Training error : 0.606701
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(svm_linear_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  29.4
#> 2  30.9
#> 3  21.7
#> 4  47.1
#> 5  36.4
#> 6  33.4
#> 7  34.2
#> 8  43.2
```
:::

## `LiblineaR` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
svm_linear_spec <- svm_linear() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because LiblineaR is the default.
  set_mode("regression")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
svm_linear_fit <- svm_linear_spec |>
  fit(strength ~ ., data = reg_train)
svm_linear_fit
#> parsnip model object
#> 
#> $TypeDetail
#> [1] "L2-regularized L2-loss support vector regression primal (L2R_L2LOSS_SVR)"
#> 
#> $Type
#> [1] 11
#> 
#> $W
#>        cement      age     Bias
#> [1,] 8.665447 5.486263 33.34299
#> 
#> $Bias
#> [1] 1
#> 
#> $NbClass
#> [1] 2
#> 
#> attr(,"class")
#> [1] "LiblineaR"
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(svm_linear_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  31.9
#> 2  30.1
#> 3  21.5
#> 4  50.9
#> 5  39.9
#> 6  35.0
#> 7  36.0
#> 8  48.3
```
:::

:::

### Support Vector Machine (Polynomial Kernel) (`svm_poly()`)

:::{.panel-tabset}

## `kernlab`

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
svm_poly_spec <- svm_poly() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because kernlab is the default.
  set_mode("regression")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
svm_poly_fit <- svm_poly_spec |>
  fit(strength ~ ., data = reg_train)
#>  Setting default kernel parameters
svm_poly_fit
#> parsnip model object
#> 
#> Support Vector Machine object of class "ksvm" 
#> 
#> SV type: eps-svr  (regression) 
#>  parameter : epsilon = 0.1  cost C = 1 
#> 
#> Polynomial kernel function. 
#>  Hyperparameters : degree =  1  scale =  1  offset =  1 
#> 
#> Number of Support Vectors : 85 
#> 
#> Objective Function Value : -47.4495 
#> Training error : 0.606702
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(svm_poly_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  29.4
#> 2  30.9
#> 3  21.7
#> 4  47.1
#> 5  36.4
#> 6  33.4
#> 7  34.2
#> 8  43.2
```
:::

:::

### Support Vector Machine (Radial Basis Function Kernel) (`svm_rbf()`)

:::{.panel-tabset}

## `kernlab`

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
svm_rbf_spec <- svm_rbf() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because kernlab is the default.
  set_mode("regression")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
svm_rbf_fit <- svm_rbf_spec |>
  fit(strength ~ ., data = reg_train)
svm_rbf_fit
#> parsnip model object
#> 
#> Support Vector Machine object of class "ksvm" 
#> 
#> SV type: eps-svr  (regression) 
#>  parameter : epsilon = 0.1  cost C = 1 
#> 
#> Gaussian Radial Basis kernel function. 
#>  Hyperparameter : sigma =  0.850174270140177 
#> 
#> Number of Support Vectors : 79 
#> 
#> Objective Function Value : -33.0277 
#> Training error : 0.28361
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(svm_rbf_fit, new_data = reg_test)
#> # A tibble: 8 × 1
#>   .pred
#>   <dbl>
#> 1  20.0
#> 2  41.3
#> 3  26.0
#> 4  53.5
#> 5  35.2
#> 6  34.7
#> 7  36.2
#> 8  42.3
```
:::

:::

# Censored Regression Models

## Example data

Let's simulate a data set using the prodlim and survival packages: 

::: {.cell layout-align="center"}

```{.r .cell-code}
library(survival)
#> 
#> Attaching package: 'survival'
#> The following object is masked from 'package:future':
#> 
#>     cluster
library(prodlim)

set.seed(1000)
cns_data <- 
  SimSurv(250) |> 
  mutate(event_time = Surv(time, event)) |> 
  select(event_time, X1, X2)

cns_split <- initial_split(cns_data, prop = 0.98)
cns_split
#> <Training/Testing/Total>
#> <245/5/250>
cns_train <- training(cns_split)
cns_test <- testing(cns_split)
```
:::

For some types of predictions, we need the _evaluation time(s)_ for the predictions. We'll use these three times to demonstrate: 

::: {.cell layout-align="center"}

```{.r .cell-code}
eval_times <- c(1, 3, 5)
```
:::

## Models

### Bagged Decision Trees (`bag_tree()`) 

:::{.panel-tabset}

## `rpart` 

This engine requires the censored extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(censored)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
bag_tree_spec <- bag_tree() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because rpart is the default.
  set_mode("censored regression")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
bag_tree_fit <- bag_tree_spec |>
  fit(event_time ~ ., data = cns_train)
bag_tree_fit
#> parsnip model object
#> 
#> 
#> Bagging survival trees with 25 bootstrap replications 
#> 
#> Call: bagging.data.frame(formula = event_time ~ ., data = data, cp = ~0, 
#>     minsplit = ~2)
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(bag_tree_fit, type = "time", new_data = cns_test)
#> # A tibble: 5 × 1
#>   .pred_time
#>        <dbl>
#> 1       5.65
#> 2       4.12
#> 3       5.03
#> 4       5.58
#> 5       4.88
predict(bag_tree_fit, type = "survival", new_data = cns_test, eval_time = eval_times)
#> # A tibble: 5 × 1
#>   .pred           
#>   <list>          
#> 1 <tibble [3 × 2]>
#> 2 <tibble [3 × 2]>
#> 3 <tibble [3 × 2]>
#> 4 <tibble [3 × 2]>
#> 5 <tibble [3 × 2]>
```
:::

Each row of the survival predictions has results for each evaluation time: 

::: {.cell layout-align="center"}

```{.r .cell-code}
bag_tree_fit |> 
  predict(type = "survival", new_data = cns_test, eval_time = eval_times) |> 
  slice(1) |> 
  pluck(".pred")
#> [[1]]
#> # A tibble: 3 × 2
#>   .eval_time .pred_survival
#>        <dbl>          <dbl>
#> 1          1          0.993
#> 2          3          0.864
#> 3          5          0.638
```
:::

:::

### Boosted Decision Trees (`boost_tree()`)

:::{.panel-tabset}

## `mboost`

This engine requires the censored extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(censored)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
boost_tree_spec <- boost_tree() |> 
  set_mode("censored regression") |> 
  set_engine("mboost")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(852)
boost_tree_fit <- boost_tree_spec |>
  fit(event_time ~ ., data = cns_train)
boost_tree_fit
#> parsnip model object
#> 
#> 
#> 	 Model-based Boosting
#> 
#> Call:
#> mboost::blackboost(formula = formula, data = data, family = family,     control = mboost::boost_control(), tree_controls = partykit::ctree_control(teststat = "quadratic",         testtype = "Teststatistic", mincriterion = 0, minsplit = 10,         minbucket = 4, maxdepth = 2, saveinfo = FALSE))
#> 
#> 
#> 	 Cox Partial Likelihood 
#> 
#> Loss function:  
#> 
#> Number of boosting iterations: mstop = 100 
#> Step size:  0.1 
#> Offset:  0 
#> Number of baselearners:  1
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(boost_tree_fit, type = "time", new_data = cns_test)
#> # A tibble: 5 × 1
#>   .pred_time
#>        <dbl>
#> 1       6.51
#> 2       3.92
#> 3       4.51
#> 4       7.17
#> 5       4.51
predict(boost_tree_fit, type = "survival", new_data = cns_test, eval_time = eval_times)
#> # A tibble: 5 × 1
#>   .pred           
#>   <list>          
#> 1 <tibble [3 × 2]>
#> 2 <tibble [3 × 2]>
#> 3 <tibble [3 × 2]>
#> 4 <tibble [3 × 2]>
#> 5 <tibble [3 × 2]>
predict(boost_tree_fit, type = "linear_pred", new_data = cns_test)
#> # A tibble: 5 × 1
#>   .pred_linear_pred
#>               <dbl>
#> 1           0.00839
#> 2          -1.14   
#> 3          -0.823  
#> 4           0.229  
#> 5          -0.823
```
:::

Each row of the survival predictions has results for each evaluation time: 

::: {.cell layout-align="center"}

```{.r .cell-code}
boost_tree_fit |> 
  predict(type = "survival", new_data = cns_test, eval_time = eval_times) |> 
  slice(1) |> 
  pluck(".pred")
#> [[1]]
#> # A tibble: 3 × 2
#>   .eval_time .pred_survival
#>        <dbl>          <dbl>
#> 1          1          0.982
#> 2          3          0.877
#> 3          5          0.657
```
:::

:::

### Decision Tree (`decision_tree()`)

:::{.panel-tabset}

## `rpart`

This engine requires the censored extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(censored)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
decision_tree_spec <- decision_tree() |>
  # We need to set the mode since this model works with multiple modes.
  # We don't need to set the engine because rpart is the default.
  set_mode("censored regression")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
decision_tree_fit <- decision_tree_spec |>
  fit(event_time ~ ., data = cns_train)
decision_tree_fit
#> parsnip model object
#> 
#> $rpart
#> n= 245 
#> 
#> node), split, n, deviance, yval
#>       * denotes terminal node
#> 
#>  1) root 245 329.03530 1.0000000  
#>    2) X2< -0.09937043 110 119.05180 0.5464982  
#>      4) X2< -0.9419799 41  42.43138 0.3153769  
#>        8) X1< 0.5 20  12.93725 0.1541742 *
#>        9) X1>=0.5 21  23.29519 0.5656502 *
#>      5) X2>=-0.9419799 69  67.76223 0.7336317 *
#>    3) X2>=-0.09937043 135 157.14990 1.7319010  
#>      6) X1< 0.5 79  66.30972 1.2572690 *
#>      7) X1>=0.5 56  69.62652 3.0428230  
#>       14) X2< 1.222057 44  40.33335 2.5072040 *
#>       15) X2>=1.222057 12  17.95790 6.3934130 *
#> 
#> $survfit
#> 
#> Call: prodlim::prodlim(formula = form, data = data)
#> Stratified Kaplan-Meier estimator for the conditional event time survival function
#> Discrete predictor variable: rpartFactor (0.154174164904031, 0.565650228981439, 0.733631734872791, 1.25726850344687, 2.50720371146533, 6.39341334321542)
#> 
#> Right-censored response of a survival model
#> 
#> No.Observations: 245 
#> 
#> Pattern:
#>                 Freq
#>  event          161 
#>  right.censored 84  
#> 
#> $levels
#> [1] "0.154174164904031" "0.565650228981439" "0.733631734872791"
#> [4] "1.25726850344687"  "2.50720371146533"  "6.39341334321542" 
#> 
#> attr(,"class")
#> [1] "pecRpart"
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(decision_tree_fit, type = "time", new_data = cns_test)
#> # A tibble: 5 × 1
#>   .pred_time
#>        <dbl>
#> 1       1.26
#> 2       2.51
#> 3       1.26
#> 4       1.26
#> 5       1.26
predict(decision_tree_fit, type = "survival", new_data = cns_test, eval_time = eval_times)
#> # A tibble: 5 × 1
#>   .pred           
#>   <list>          
#> 1 <tibble [3 × 2]>
#> 2 <tibble [3 × 2]>
#> 3 <tibble [3 × 2]>
#> 4 <tibble [3 × 2]>
#> 5 <tibble [3 × 2]>
```
:::

Each row of the survival predictions has results for each evaluation time: 

::: {.cell layout-align="center"}

```{.r .cell-code}
decision_tree_fit |> 
  predict(type = "survival", new_data = cns_test, eval_time = eval_times) |> 
  slice(1) |> 
  pluck(".pred")
#> [[1]]
#> # A tibble: 3 × 2
#>   .eval_time .pred_survival
#>        <dbl>          <dbl>
#> 1          1          0.987
#> 2          3          0.854
#> 3          5          0.634
```
:::

## `partykit` 

This engine requires the censored extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(censored)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
decision_tree_spec <- decision_tree() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("censored regression") |>
  set_engine("partykit")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
decision_tree_fit <- decision_tree_spec |>
  fit(event_time ~ ., data = cns_train)
decision_tree_fit
#> parsnip model object
#> 
#> 
#> Model formula:
#> event_time ~ X1 + X2
#> 
#> Fitted party:
#> [1] root
#> |   [2] X2 <= -0.36159
#> |   |   [3] X1 <= 0: 13.804 (n = 41)
#> |   |   [4] X1 > 0: 8.073 (n = 47)
#> |   [5] X2 > -0.36159
#> |   |   [6] X1 <= 0: 6.274 (n = 89)
#> |   |   [7] X1 > 0
#> |   |   |   [8] X2 <= 0.56098: 5.111 (n = 39)
#> |   |   |   [9] X2 > 0.56098: 2.713 (n = 29)
#> 
#> Number of inner nodes:    4
#> Number of terminal nodes: 5
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(decision_tree_fit, type = "time", new_data = cns_test)
#> # A tibble: 5 × 1
#>   .pred_time
#>        <dbl>
#> 1       6.27
#> 2       5.11
#> 3       6.27
#> 4       6.27
#> 5       6.27
predict(decision_tree_fit, type = "survival", new_data = cns_test, eval_time = eval_times)
#> # A tibble: 5 × 1
#>   .pred           
#>   <list>          
#> 1 <tibble [3 × 2]>
#> 2 <tibble [3 × 2]>
#> 3 <tibble [3 × 2]>
#> 4 <tibble [3 × 2]>
#> 5 <tibble [3 × 2]>
```
:::

Each row of the survival predictions has results for each evaluation time: 

::: {.cell layout-align="center"}

```{.r .cell-code}
decision_tree_fit |> 
  predict(type = "survival", new_data = cns_test, eval_time = eval_times) |> 
  slice(1) |> 
  pluck(".pred")
#> [[1]]
#> # A tibble: 3 × 2
#>   .eval_time .pred_survival
#>        <dbl>          <dbl>
#> 1          1          0.989
#> 2          3          0.871
#> 3          5          0.649
```
:::

:::

### Proportional Hazards (`proportional_hazards()`)

:::{.panel-tabset}

## `survival` 

This engine requires the censored extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(censored)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
# This model only works with a single mode, so we don't need to set the mode.
# We don't need to set the engine because survival is the default.
proportional_hazards_spec <- proportional_hazards()
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
proportional_hazards_fit <- proportional_hazards_spec |>
  fit(event_time ~ ., data = cns_train)
proportional_hazards_fit
#> parsnip model object
#> 
#> Call:
#> survival::coxph(formula = event_time ~ ., data = data, model = TRUE, 
#>     x = TRUE)
#> 
#>       coef exp(coef) se(coef)     z        p
#> X1 0.99547   2.70599  0.16799 5.926 3.11e-09
#> X2 0.91398   2.49422  0.09566 9.555  < 2e-16
#> 
#> Likelihood ratio test=106.8  on 2 df, p=< 2.2e-16
#> n= 245, number of events= 161
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(proportional_hazards_fit, type = "time", new_data = cns_test)
#> # A tibble: 5 × 1
#>   .pred_time
#>        <dbl>
#> 1       7.87
#> 2       4.16
#> 3       4.62
#> 4       5.19
#> 5       4.41
predict(proportional_hazards_fit, type = "survival", new_data = cns_test, eval_time = eval_times)
#> # A tibble: 5 × 1
#>   .pred           
#>   <list>          
#> 1 <tibble [3 × 2]>
#> 2 <tibble [3 × 2]>
#> 3 <tibble [3 × 2]>
#> 4 <tibble [3 × 2]>
#> 5 <tibble [3 × 2]>
predict(proportional_hazards_fit, type = "linear_pred", new_data = cns_test)
#> # A tibble: 5 × 1
#>   .pred_linear_pred
#>               <dbl>
#> 1            -0.111
#> 2            -1.49 
#> 3            -1.27 
#> 4            -1.02 
#> 5            -1.37
```
:::

Each row of the survival predictions has results for each evaluation time: 

::: {.cell layout-align="center"}

```{.r .cell-code}
proportional_hazards_fit |> 
  predict(type = "survival", new_data = cns_test, eval_time = eval_times) |> 
  slice(1) |> 
  pluck(".pred")
#> [[1]]
#> # A tibble: 3 × 2
#>   .eval_time .pred_survival
#>        <dbl>          <dbl>
#> 1          1          0.985
#> 2          3          0.909
#> 3          5          0.750
```
:::

## `glmnet` 

This engine requires the censored extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(censored)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
proportional_hazards_spec <- proportional_hazards(penalty = 0.01) |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("glmnet")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
proportional_hazards_fit <- proportional_hazards_spec |>
  fit(event_time ~ ., data = cns_train)
proportional_hazards_fit
#> parsnip model object
#> 
#> 
#> Call:  glmnet::glmnet(x = data_obj$x, y = data_obj$y, family = "cox",      weights = weights, alpha = alpha, lambda = lambda) 
#> 
#>    Df %Dev  Lambda
#> 1   0 0.00 0.39700
#> 2   1 0.82 0.36170
#> 3   1 1.51 0.32960
#> 4   1 2.07 0.30030
#> 5   1 2.54 0.27360
#> 6   1 2.94 0.24930
#> 7   2 3.28 0.22720
#> 8   2 3.95 0.20700
#> 9   2 4.50 0.18860
#> 10  2 4.95 0.17180
#> 11  2 5.33 0.15660
#> 12  2 5.65 0.14270
#> 13  2 5.91 0.13000
#> 14  2 6.13 0.11840
#> 15  2 6.31 0.10790
#> 16  2 6.46 0.09833
#> 17  2 6.58 0.08960
#> 18  2 6.69 0.08164
#> 19  2 6.77 0.07439
#> 20  2 6.85 0.06778
#> 21  2 6.91 0.06176
#> 22  2 6.96 0.05627
#> 23  2 7.00 0.05127
#> 24  2 7.03 0.04672
#> 25  2 7.06 0.04257
#> 26  2 7.08 0.03879
#> 27  2 7.10 0.03534
#> 28  2 7.12 0.03220
#> 29  2 7.13 0.02934
#> 30  2 7.14 0.02673
#> 31  2 7.15 0.02436
#> 32  2 7.16 0.02219
#> 33  2 7.17 0.02022
#> 34  2 7.17 0.01843
#> 35  2 7.18 0.01679
#> 36  2 7.18 0.01530
#> 37  2 7.18 0.01394
#> 38  2 7.19 0.01270
#> 39  2 7.19 0.01157
#> 40  2 7.19 0.01054
#> 41  2 7.19 0.00961
#> 42  2 7.19 0.00875
#> The training data has been saved for prediction.
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(proportional_hazards_fit, type = "time", new_data = cns_test)
#> # A tibble: 5 × 1
#>   .pred_time
#>        <dbl>
#> 1       7.80
#> 2       4.21
#> 3       4.63
#> 4       5.18
#> 5       4.42
predict(proportional_hazards_fit, type = "survival", new_data = cns_test, eval_time = eval_times)
#> # A tibble: 5 × 1
#>   .pred           
#>   <list>          
#> 1 <tibble [3 × 2]>
#> 2 <tibble [3 × 2]>
#> 3 <tibble [3 × 2]>
#> 4 <tibble [3 × 2]>
#> 5 <tibble [3 × 2]>
predict(proportional_hazards_fit, type = "linear_pred", new_data = cns_test)
#> # A tibble: 5 × 1
#>   .pred_linear_pred
#>               <dbl>
#> 1            -0.108
#> 2            -1.43 
#> 3            -1.23 
#> 4            -0.993
#> 5            -1.33
```
:::

Each row of the survival predictions has results for each evaluation time: 

::: {.cell layout-align="center"}

```{.r .cell-code}
proportional_hazards_fit |> 
  predict(type = "survival", new_data = cns_test, eval_time = eval_times) |> 
  slice(1) |> 
  pluck(".pred")
#> [[1]]
#> # A tibble: 3 × 2
#>   .eval_time .pred_survival
#>        <dbl>          <dbl>
#> 1          1          0.984
#> 2          3          0.906
#> 3          5          0.743
```
:::

:::

### Random Forests (`rand_forest()`)

:::{.panel-tabset}

## `aorsf`

This engine requires the censored extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(censored)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
rand_forest_spec <- rand_forest() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("censored regression") |>
  set_engine("aorsf")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(2)
rand_forest_fit <- rand_forest_spec |>
  fit(event_time ~ ., data = cns_train)
rand_forest_fit
#> parsnip model object
#> 
#> ---------- Oblique random survival forest
#> 
#>      Linear combinations: Accelerated Cox regression
#>           N observations: 245
#>                 N events: 161
#>                  N trees: 500
#>       N predictors total: 2
#>    N predictors per node: 2
#>  Average leaves per tree: 12.85
#> Min observations in leaf: 5
#>       Min events in leaf: 1
#>           OOB stat value: 0.70
#>            OOB stat type: Harrell's C-index
#>      Variable importance: anova
#> 
#> -----------------------------------------
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(rand_forest_fit, type = "time", new_data = cns_test)
#> # A tibble: 5 × 1
#>   .pred_time
#>        <dbl>
#> 1       5.93
#> 2       3.85
#> 3       4.41
#> 4       5.43
#> 5       4.34
predict(rand_forest_fit, type = "survival", new_data = cns_test, eval_time = eval_times)
#> # A tibble: 5 × 1
#>   .pred           
#>   <list>          
#> 1 <tibble [3 × 2]>
#> 2 <tibble [3 × 2]>
#> 3 <tibble [3 × 2]>
#> 4 <tibble [3 × 2]>
#> 5 <tibble [3 × 2]>
```
:::

Each row of the survival predictions has results for each evaluation time: 

::: {.cell layout-align="center"}

```{.r .cell-code}
rand_forest_fit |> 
  predict(type = "survival", new_data = cns_test, eval_time = eval_times) |> 
  slice(1) |> 
  pluck(".pred")
#> [[1]]
#> # A tibble: 3 × 2
#>   .eval_time .pred_survival
#>        <dbl>          <dbl>
#> 1          1          0.999
#> 2          3          0.873
#> 3          5          0.627
```
:::

## `partykit` 

This engine requires the censored extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(censored)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
rand_forest_spec <- rand_forest() |>
  # We need to set the mode since this model works with multiple modes.
  set_mode("censored regression") |>
  set_engine("partykit")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(89)
rand_forest_fit <- rand_forest_spec |>
  fit(event_time ~ ., data = cns_train)
```
:::

The print method has a lot of output: 

<details>

::: {.cell layout-align="center"}

```{.r .cell-code}
capture.output(print(rand_forest_fit))[1:100] |> cat(sep = "\n")
#> parsnip model object
#> 
#> $nodes
#> $nodes[[1]]
#> [1] root
#> |   [2] V3 <= -0.16072
#> |   |   [3] V2 <= 0
#> |   |   |   [4] V3 <= -1.68226 *
#> |   |   |   [5] V3 > -1.68226
#> |   |   |   |   [6] V3 <= -0.65952 *
#> |   |   |   |   [7] V3 > -0.65952 *
#> |   |   [8] V2 > 0
#> |   |   |   [9] V3 <= -0.98243 *
#> |   |   |   [10] V3 > -0.98243
#> |   |   |   |   [11] V3 <= -0.67216 *
#> |   |   |   |   [12] V3 > -0.67216 *
#> |   [13] V3 > -0.16072
#> |   |   [14] V2 <= 0
#> |   |   |   [15] V3 <= 0.95981
#> |   |   |   |   [16] V3 <= 0.3117
#> |   |   |   |   |   [17] V3 <= 0.09688 *
#> |   |   |   |   |   [18] V3 > 0.09688 *
#> |   |   |   |   [19] V3 > 0.3117
#> |   |   |   |   |   [20] V3 <= 0.40845 *
#> |   |   |   |   |   [21] V3 > 0.40845 *
#> |   |   |   [22] V3 > 0.95981 *
#> |   |   [23] V2 > 0
#> |   |   |   [24] V3 <= 0.56098 *
#> |   |   |   [25] V3 > 0.56098 *
#> 
#> $nodes[[2]]
#> [1] root
#> |   [2] V3 <= -0.36618
#> |   |   [3] V2 <= 0
#> |   |   |   [4] V3 <= -1.19881 *
#> |   |   |   [5] V3 > -1.19881 *
#> |   |   [6] V2 > 0
#> |   |   |   [7] V3 <= -1.18263 *
#> |   |   |   [8] V3 > -1.18263
#> |   |   |   |   [9] V3 <= -0.55449 *
#> |   |   |   |   [10] V3 > -0.55449 *
#> |   [11] V3 > -0.36618
#> |   |   [12] V2 <= 0
#> |   |   |   [13] V3 <= 0.3117
#> |   |   |   |   [14] V3 <= -0.01851 *
#> |   |   |   |   [15] V3 > -0.01851 *
#> |   |   |   [16] V3 > 0.3117
#> |   |   |   |   [17] V3 <= 0.85976 *
#> |   |   |   |   [18] V3 > 0.85976 *
#> |   |   [19] V2 > 0
#> |   |   |   [20] V3 <= -0.04369 *
#> |   |   |   [21] V3 > -0.04369
#> |   |   |   |   [22] V3 <= 0.56098 *
#> |   |   |   |   [23] V3 > 0.56098
#> |   |   |   |   |   [24] V3 <= 1.22094 *
#> |   |   |   |   |   [25] V3 > 1.22094 *
#> 
#> $nodes[[3]]
#> [1] root
#> |   [2] V3 <= -0.46092
#> |   |   [3] V2 <= 0
#> |   |   |   [4] V3 <= -1.65465 *
#> |   |   |   [5] V3 > -1.65465 *
#> |   |   [6] V2 > 0
#> |   |   |   [7] V3 <= -1.36941 *
#> |   |   |   [8] V3 > -1.36941
#> |   |   |   |   [9] V3 <= -0.83366 *
#> |   |   |   |   [10] V3 > -0.83366 *
#> |   [11] V3 > -0.46092
#> |   |   [12] V2 <= 0
#> |   |   |   [13] V3 <= -0.01851 *
#> |   |   |   [14] V3 > -0.01851
#> |   |   |   |   [15] V3 <= 0.22967 *
#> |   |   |   |   [16] V3 > 0.22967
#> |   |   |   |   |   [17] V3 <= 0.95368
#> |   |   |   |   |   |   [18] V3 <= 0.68292 *
#> |   |   |   |   |   |   [19] V3 > 0.68292 *
#> |   |   |   |   |   [20] V3 > 0.95368 *
#> |   |   [21] V2 > 0
#> |   |   |   [22] V3 <= 0.15595 *
#> |   |   |   [23] V3 > 0.15595
#> |   |   |   |   [24] V3 <= 0.51117 *
#> |   |   |   |   [25] V3 > 0.51117 *
#> 
#> $nodes[[4]]
#> [1] root
#> |   [2] V3 <= -0.10421
#> |   |   [3] V2 <= 0
#> |   |   |   [4] V3 <= -0.96818 *
#> |   |   |   [5] V3 > -0.96818
#> |   |   |   |   [6] V3 <= -0.64682 *
#> |   |   |   |   [7] V3 > -0.64682 *
#> |   |   [8] V2 > 0
#> |   |   |   [9] V3 <= -0.83366 *
#> |   |   |   [10] V3 > -0.83366 *
#> |   [11] V3 > -0.10421
#> |   |   [12] V2 <= 0
#> |   |   |   [13] V3 <= 0.14347 *
#> |   |   |   [14] V3 > 0.14347
#> |   |   |   |   [15] V3 <= 1.20345
```
:::

</details>

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(rand_forest_fit, type = "time", new_data = cns_test)
#> # A tibble: 5 × 1
#>   .pred_time
#>        <dbl>
#> 1       5.22
#> 2       4.12
#> 3       3.87
#> 4       4.82
#> 5       3.87
predict(rand_forest_fit, type = "survival", new_data = cns_test, eval_time = eval_times)
#> # A tibble: 5 × 1
#>   .pred           
#>   <list>          
#> 1 <tibble [3 × 2]>
#> 2 <tibble [3 × 2]>
#> 3 <tibble [3 × 2]>
#> 4 <tibble [3 × 2]>
#> 5 <tibble [3 × 2]>
```
:::

Each row of the survival predictions has results for each evaluation time: 

::: {.cell layout-align="center"}

```{.r .cell-code}
rand_forest_fit |> 
  predict(type = "survival", new_data = cns_test, eval_time = eval_times) |> 
  slice(1) |> 
  pluck(".pred")
#> [[1]]
#> # A tibble: 3 × 2
#>   .eval_time .pred_survival
#>        <dbl>          <dbl>
#> 1          1          1    
#> 2          3          0.870
#> 3          5          0.594
```
:::

:::

### Parametric Survival Models (`survival_reg()`)

:::{.panel-tabset}

## `survival` 

This engine requires the censored extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(censored)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
# This model only works with a single mode, so we don't need to set the mode.
# We don't need to set the engine because survival is the default.
survival_reg_spec <- survival_reg()
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
survival_reg_fit <- survival_reg_spec |>
  fit(event_time ~ ., data = cns_train)
survival_reg_fit
#> parsnip model object
#> 
#> Call:
#> survival::survreg(formula = event_time ~ ., data = data, model = TRUE)
#> 
#> Coefficients:
#> (Intercept)          X1          X2 
#>   2.2351722  -0.4648296  -0.4222887 
#> 
#> Scale= 0.4728442 
#> 
#> Loglik(model)= -427.4   Loglik(intercept only)= -481.3
#> 	Chisq= 107.73 on 2 degrees of freedom, p= <2e-16 
#> n= 245
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(survival_reg_fit, type = "time", new_data = cns_test)
#> # A tibble: 5 × 1
#>   .pred_time
#>        <dbl>
#> 1       8.88
#> 2       4.67
#> 3       5.20
#> 4       5.83
#> 5       4.97
predict(survival_reg_fit, type = "survival", new_data = cns_test, eval_time = eval_times)
#> # A tibble: 5 × 1
#>   .pred           
#>   <list>          
#> 1 <tibble [3 × 2]>
#> 2 <tibble [3 × 2]>
#> 3 <tibble [3 × 2]>
#> 4 <tibble [3 × 2]>
#> 5 <tibble [3 × 2]>
predict(survival_reg_fit, type = "hazard", new_data = cns_test, eval_time = eval_times)
#> # A tibble: 5 × 1
#>   .pred           
#>   <list>          
#> 1 <tibble [3 × 2]>
#> 2 <tibble [3 × 2]>
#> 3 <tibble [3 × 2]>
#> 4 <tibble [3 × 2]>
#> 5 <tibble [3 × 2]>
predict(survival_reg_fit, type = "linear_pred", new_data = cns_test)
#> # A tibble: 5 × 1
#>   .pred_linear_pred
#>               <dbl>
#> 1              2.18
#> 2              1.54
#> 3              1.65
#> 4              1.76
#> 5              1.60
predict(survival_reg_fit, type = "quantile", new_data = cns_test)
#> # A tibble: 5 × 1
#>   .pred_quantile
#>        <qtls(9)>
#> 1         [7.47]
#> 2         [3.92]
#> 3         [4.37]
#> 4          [4.9]
#> 5         [4.18]
```
:::

Each row of the survival predictions has results for each evaluation time: 

::: {.cell layout-align="center"}

```{.r .cell-code}
survival_reg_fit |> 
  predict(type = "survival", new_data = cns_test, eval_time = eval_times) |> 
  slice(1) |> 
  pluck(".pred")
#> [[1]]
#> # A tibble: 3 × 2
#>   .eval_time .pred_survival
#>        <dbl>          <dbl>
#> 1          1          0.990
#> 2          3          0.904
#> 3          5          0.743
```
:::

## `flexsurv` 

This engine requires the censored extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(censored)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
survival_reg_spec <- survival_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("flexsurv")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
survival_reg_fit <- survival_reg_spec |>
  fit(event_time ~ ., data = cns_train)
survival_reg_fit
#> parsnip model object
#> 
#> Call:
#> flexsurv::flexsurvreg(formula = event_time ~ ., data = data, 
#>     dist = "weibull")
#> 
#> Estimates: 
#>        data mean  est       L95%      U95%      se        exp(est)  L95%    
#> shape        NA    2.11486   1.87774   2.38192   0.12832        NA        NA
#> scale        NA    9.34809   8.38852  10.41743   0.51658        NA        NA
#> X1      0.46939   -0.46483  -0.61347  -0.31619   0.07584   0.62824   0.54147
#> X2     -0.00874   -0.42229  -0.50641  -0.33817   0.04292   0.65554   0.60266
#>        U95%    
#> shape        NA
#> scale        NA
#> X1      0.72892
#> X2      0.71307
#> 
#> N = 245,  Events: 161,  Censored: 84
#> Total time at risk: 1388.951
#> Log-likelihood = -427.4387, df = 4
#> AIC = 862.8774
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(survival_reg_fit, type = "time", new_data = cns_test)
#> # A tibble: 5 × 1
#>   .pred_time
#>        <dbl>
#> 1       7.87
#> 2       4.13
#> 3       4.61
#> 4       5.16
#> 5       4.40
predict(survival_reg_fit, type = "survival", new_data = cns_test, eval_time = eval_times)
#> # A tibble: 5 × 1
#>   .pred           
#>   <list>          
#> 1 <tibble [3 × 2]>
#> 2 <tibble [3 × 2]>
#> 3 <tibble [3 × 2]>
#> 4 <tibble [3 × 2]>
#> 5 <tibble [3 × 2]>
predict(survival_reg_fit, type = "hazard", new_data = cns_test, eval_time = eval_times)
#> # A tibble: 5 × 1
#>   .pred           
#>   <list>          
#> 1 <tibble [3 × 2]>
#> 2 <tibble [3 × 2]>
#> 3 <tibble [3 × 2]>
#> 4 <tibble [3 × 2]>
#> 5 <tibble [3 × 2]>
predict(survival_reg_fit, type = "linear_pred", new_data = cns_test)
#> # A tibble: 5 × 1
#>   .pred_linear_pred
#>               <dbl>
#> 1              2.18
#> 2              1.54
#> 3              1.65
#> 4              1.76
#> 5              1.60
predict(survival_reg_fit, type = "quantile", new_data = cns_test)
#> # A tibble: 5 × 1
#>   .pred_quantile
#>        <qtls(9)>
#> 1         [7.47]
#> 2         [3.92]
#> 3         [4.37]
#> 4          [4.9]
#> 5         [4.18]
```
:::

Each row of the survival predictions has results for each evaluation time: 

::: {.cell layout-align="center"}

```{.r .cell-code}
survival_reg_fit |> 
  predict(type = "survival", new_data = cns_test, eval_time = eval_times) |> 
  slice(1) |> 
  pluck(".pred")
#> [[1]]
#> # A tibble: 3 × 2
#>   .eval_time .pred_survival
#>        <dbl>          <dbl>
#> 1          1          0.990
#> 2          3          0.904
#> 3          5          0.743
```
:::

## `flexsurvspline` 

This engine requires the censored extension package, so let's load this first:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(censored)
```
:::

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
survival_reg_spec <- survival_reg() |> 
  # This model only works with a single mode, so we don't need to set the mode.
  set_engine("flexsurvspline")
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
survival_reg_fit <- survival_reg_spec |>
  fit(event_time ~ ., data = cns_train)
survival_reg_fit
#> parsnip model object
#> 
#> Call:
#> flexsurv::flexsurvspline(formula = event_time ~ ., data = data)
#> 
#> Estimates: 
#>         data mean  est       L95%      U95%      se        exp(est)  L95%    
#> gamma0        NA   -4.72712  -5.31772  -4.13651   0.30134        NA        NA
#> gamma1        NA    2.11487   1.86338   2.36637   0.12832        NA        NA
#> X1       0.46939    0.98305   0.65928   1.30683   0.16519   2.67261   1.93340
#> X2      -0.00874    0.89308   0.70943   1.07673   0.09370   2.44265   2.03283
#>         U95%    
#> gamma0        NA
#> gamma1        NA
#> X1       3.69444
#> X2       2.93508
#> 
#> N = 245,  Events: 161,  Censored: 84
#> Total time at risk: 1388.951
#> Log-likelihood = -427.4387, df = 4
#> AIC = 862.8774
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(survival_reg_fit, type = "time", new_data = cns_test)
#> # A tibble: 5 × 1
#>   .pred_time
#>        <dbl>
#> 1       7.87
#> 2       4.13
#> 3       4.61
#> 4       5.16
#> 5       4.40
predict(survival_reg_fit, type = "survival", new_data = cns_test, eval_time = eval_times)
#> # A tibble: 5 × 1
#>   .pred           
#>   <list>          
#> 1 <tibble [3 × 2]>
#> 2 <tibble [3 × 2]>
#> 3 <tibble [3 × 2]>
#> 4 <tibble [3 × 2]>
#> 5 <tibble [3 × 2]>
predict(survival_reg_fit, type = "hazard", new_data = cns_test, eval_time = eval_times)
#> # A tibble: 5 × 1
#>   .pred           
#>   <list>          
#> 1 <tibble [3 × 2]>
#> 2 <tibble [3 × 2]>
#> 3 <tibble [3 × 2]>
#> 4 <tibble [3 × 2]>
#> 5 <tibble [3 × 2]>
predict(survival_reg_fit, type = "linear_pred", new_data = cns_test)
#> # A tibble: 5 × 1
#>   .pred_linear_pred
#>               <dbl>
#> 1             -4.62
#> 2             -3.26
#> 3             -3.49
#> 4             -3.73
#> 5             -3.39
predict(survival_reg_fit, type = "quantile", new_data = cns_test)
#> # A tibble: 5 × 1
#>   .pred_quantile
#>        <qtls(9)>
#> 1         [7.47]
#> 2         [3.92]
#> 3         [4.37]
#> 4          [4.9]
#> 5         [4.18]
```
:::

Each row of the survival predictions has results for each evaluation time: 

::: {.cell layout-align="center"}

```{.r .cell-code}
survival_reg_fit |> 
  predict(type = "survival", new_data = cns_test, eval_time = eval_times) |> 
  slice(1) |> 
  pluck(".pred")
#> [[1]]
#> # A tibble: 3 × 2
#>   .eval_time .pred_survival
#>        <dbl>          <dbl>
#> 1          1          0.990
#> 2          3          0.904
#> 3          5          0.743
```
:::

:::

# Quantile Regression Models

## Example data

To demonstrate quantile regression, let's make a larger version of our regression data: 

::: {.cell layout-align="center"}

```{.r .cell-code}
set.seed(938)
qnt_split <-
  modeldata::concrete |> 
  slice_sample(n = 100) |> 
  select(strength = compressive_strength, cement, age) |> 
  initial_split(prop = 0.95, strata = strength)
qnt_split
#> <Training/Testing/Total>
#> <92/8/100>

qnt_rec <- 
  recipe(strength ~ ., data = training(qnt_split)) |> 
  step_normalize(all_numeric_predictors()) |> 
  prep()

qnt_train <- bake(qnt_rec, new_data = NULL)
qnt_test <- bake(qnt_rec, new_data = testing(qnt_split))
```
:::

We'll also predict these quantile levels: 

::: {.cell layout-align="center"}

```{.r .cell-code}
qnt_lvls <- (1:3) / 4
```
:::

## Models

### Linear Regression (`linear_reg()`) 

:::{.panel-tabset}

## `quantreg` 

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_spec <- linear_reg() |> 
  set_engine("quantreg") |> 
  set_mode("quantile regression", quantile_levels = qnt_lvls)
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_fit <- linear_reg_spec |>
  fit(strength ~ ., data = qnt_train)
linear_reg_fit
#> parsnip model object
#> 
#> Call:
#> quantreg::rq(formula = strength ~ ., tau = quantile_levels, data = data)
#> 
#> Coefficients:
#>             tau= 0.25 tau= 0.50 tau= 0.75
#> (Intercept) 23.498029 33.265428 42.046031
#> cement       6.635233  7.955658  8.181235
#> age          5.566668  9.514832  7.110702
#> 
#> Degrees of freedom: 92 total; 89 residual
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(linear_reg_fit, type = "quantile", new_data = qnt_test)
#> # A tibble: 8 × 1
#>   .pred_quantile
#>        <qtls(3)>
#> 1         [29.2]
#> 2         [31.5]
#> 3         [21.4]
#> 4         [48.3]
#> 5         [36.6]
#> 6         [33.8]
#> 7         [34.6]
#> 8         [43.8]
```
:::

Each row of predictions has a special vector class containing all of the quantile predictions: 

::: {.cell layout-align="center"}

```{.r .cell-code}
linear_reg_fit |> 
  predict(type = "quantile", new_data = qnt_test)|> 
  slice(1) |> 
  pluck(".pred_quantile") |> 
  # Expand the results for each quantile level by converting to a tibble
  as_tibble()
#> # A tibble: 3 × 3
#>   .pred_quantile .quantile_levels  .row
#>            <dbl>            <dbl> <int>
#> 1           21.5             0.25     1
#> 2           29.2             0.5      1
#> 3           39.5             0.75     1
```
:::

:::

### Random Forests (`rand_forest()`)

:::{.panel-tabset}

## `grf`

We create a model specification via:

::: {.cell layout-align="center"}

```{.r .cell-code}
rand_forest_spec <- rand_forest() |>
  set_engine("grf") |> 
  set_mode("quantile regression", quantile_levels = qnt_lvls)
```
:::

Now we create the model fit object:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Set the random number seed to an integer for reproducibility: 
set.seed(435)
rand_forest_fit <- rand_forest_spec |>
  fit(strength ~ ., data = qnt_train)
rand_forest_fit
#> parsnip model object
#> 
#> GRF forest object of type quantile_forest 
#> Number of trees: 2000 
#> Number of training samples: 92 
#> Variable importance: 
#>     1     2 
#> 0.454 0.546
```
:::

The holdout data can be predicted:

::: {.cell layout-align="center"}

```{.r .cell-code}
predict(rand_forest_fit, type = "quantile", new_data = qnt_test)
#> # A tibble: 8 × 1
#>   .pred_quantile
#>        <qtls(3)>
#> 1         [26.4]
#> 2         [36.2]
#> 3         [26.9]
#> 4         [43.7]
#> 5           [39]
#> 6         [35.9]
#> 7         [38.5]
#> 8         [31.8]
```
:::

Each row of predictions has a special vector class containing all of the quantile predictions: 

::: {.cell layout-align="center"}

```{.r .cell-code}
rand_forest_fit |> 
  predict(type = "quantile", new_data = qnt_test)|> 
  slice(1) |> 
  pluck(".pred_quantile") |> 
  # Expand the results for each quantile level by converting to a tibble
  as_tibble()
#> # A tibble: 3 × 3
#>   .pred_quantile .quantile_levels  .row
#>            <dbl>            <dbl> <int>
#> 1           17.2             0.25     1
#> 2           26.4             0.5      1
#> 3           39               0.75     1
```
:::

:::

## Session information {#session-info}

::: {.cell layout-align="center"}

```
#> ─ Session info ─────────────────────────────────────────────────────
#>  version  R version 4.5.2 (2025-10-31)
#>  language (EN)
#>  date     2026-02-06
#>  pandoc   3.6.3
#>  quarto   1.8.27
#> 
#> ─ Packages ─────────────────────────────────────────────────────────
#>  package         version    date (UTC) source
#>  agua            0.1.4      2024-06-04 CRAN (R 4.5.0)
#>  baguette        1.1.0      2025-01-28 CRAN (R 4.5.0)
#>  bonsai          0.4.0      2025-06-25 CRAN (R 4.5.0)
#>  broom           1.0.11     2025-12-04 CRAN (R 4.5.2)
#>  censored        0.3.3      2025-02-14 CRAN (R 4.5.0)
#>  dials           1.4.2      2025-09-04 CRAN (R 4.5.0)
#>  discrim         1.1.0      2025-12-02 CRAN (R 4.5.2)
#>  dplyr           1.2.0      2026-02-03 CRAN (R 4.5.2)
#>  ggplot2         4.0.2      2026-02-03 CRAN (R 4.5.2)
#>  HSAUR3          1.0-15     2024-08-17 CRAN (R 4.5.0)
#>  infer           1.1.0      2025-12-18 CRAN (R 4.5.2)
#>  lme4            1.1-38     2025-12-02 CRAN (R 4.5.2)
#>  multilevelmod   1.0.0      2022-06-17 CRAN (R 4.5.0)
#>  parsnip         1.4.1      2026-01-11 CRAN (R 4.5.2)
#>  plsmod          1.0.0      2022-09-06 CRAN (R 4.5.0)
#>  poissonreg      1.0.1      2022-08-22 CRAN (R 4.5.0)
#>  prodlim         2025.04.28 2025-04-28 CRAN (R 4.5.0)
#>  purrr           1.2.1      2026-01-09 CRAN (R 4.5.2)
#>  recipes         1.3.1      2025-05-21 CRAN (R 4.5.0)
#>  rlang           1.1.7      2026-01-09 CRAN (R 4.5.2)
#>  rsample         1.3.2      2026-01-30 CRAN (R 4.5.2)
#>  rules           1.0.3      2026-01-27 CRAN (R 4.5.2)
#>  sparklyr        1.9.3      2025-11-19 CRAN (R 4.5.2)
#>  survival        3.8-6      2026-01-16 CRAN (R 4.5.2)
#>  tibble          3.3.1      2026-01-11 CRAN (R 4.5.2)
#>  tidymodels      1.4.1      2025-09-08 CRAN (R 4.5.0)
#>  tune            2.0.1      2025-10-17 CRAN (R 4.5.0)
#>  workflows       1.3.0      2025-08-27 CRAN (R 4.5.0)
#>  yardstick       1.3.2      2025-01-22 CRAN (R 4.5.0)
#> 
#> ─ Python configuration ─────────────────────────────────────────────
#>  python:         /Users/hannah/.virtualenvs/r-tensorflow/bin/python
#>  libpython:      /Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/config-3.9-darwin/libpython3.9.dylib
#>  pythonhome:     /Users/hannah/.virtualenvs/r-tensorflow:/Users/hannah/.virtualenvs/r-tensorflow
#>  version:        3.9.6 (default, Dec  2 2025, 07:27:58)  [Clang 17.0.0 (clang-1700.6.3.2)]
#>  numpy:          /Users/hannah/.virtualenvs/r-tensorflow/lib/python3.9/site-packages/numpy
#>  numpy_version:  1.26.4
#>  tensorflow:     /Users/hannah/.virtualenvs/r-tensorflow/lib/python3.9/site-packages/tensorflow
#>  
#>  NOTE: Python version was forced by use_python() function
#> 
#> ────────────────────────────────────────────────────────────────────
```
:::

