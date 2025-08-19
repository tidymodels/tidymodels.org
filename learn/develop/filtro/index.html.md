---
title: "Create your own score class object"
categories:
  - developer tools
type: learn-subsection
weight: 1
description: | 
 Create a new score class object for feature selection.
toc: true
toc-depth: 3
include-after-body: ../../../resources.html
---

## Introduction

To use code in this article,  you will need to install the following packages: filtro and modeldata.

You can construct new scoring objects using `class_score()`. This article is a guide to creating new scoring objects. 

## Scoring object

All subclasses specific to the scoring method have a parent class named `class_score`. There are a few properties (attributes) for this object:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(filtro)
args(class_score)
#> function (outcome_type = c("numeric", "factor"), predictor_type = c("numeric", 
#> "factor"), case_weights = logical(0), range = integer(0), inclusive = logical(0), 
#>     fallback_value = integer(0), score_type = character(0), sorts = function() NULL, 
#>     direction = character(0), deterministic = logical(0), tuning = logical(0), 
#>     calculating_fn = function() NULL, label = character(0), packages = character(0), 
#>     results = data.frame()) 
#> NULL
```
:::

-   `outcome_type`: What types of outcome can the method handle? The options are `numeric`, `factor`, or both. 

-   `predictor_type`: What types of predictor can the method handle? The options are `numeric`, `factor`, or both. 

-   `case_weights`: Does the method accpet case weights? It is `TRUE` or `FALSE`.

-   `range`: Are there known ranges for the statistic? For example, `c(0, Inf)`, `c(0, 1)`. 

-   `inclusive`: Are these ranges inclusive at the bounds? For example, `c(FALSE, FALSE)`, `c(TRUE, TRUE)`. 

-   `fallback_value`: What is a value that can be used for the statistic so that it will never be eliminated? For example, `0`, `Inf`.

-   `score_type`: What is the column name that will be used for the statistic values? For example, `aov_pval`, `aov_fstat`. 

-   (Not used) `sorts`: How should the values be sorted (from most- to least-important)?

-   `direction`: What direction of values indicates the most important values? For example,  `maximum`, `minimize`.

-   `deterministic`: Does the fitting process use random numbers? It is `TRUE` or `FALSE`.

-   `tuning`: Does the method have tuning parameters? It is `TRUE` or `FALSE`.

-   `calculating_fn`: What function, if any, is used to estimate the values from data?

-   `label`: What label to use when printing? For example, `ANOVA p-values`, `ANOVA F-statistics`.

-   `packages`: What packages, if any, are required to train the method?

-   `results`: A slot for the results once the method is fitted. Initially, this is an empty data frame.

## Scoring object specific to the scoring method

As an example, let’s consider the ANOVA F-test filter. 

`class_score_aov` is a subclass of `class_score`. Any additional properties specific to the implementation can be added in the subclass. For example: 

::: {.cell layout-align="center"}

```{.r .cell-code}
# Create a subclass named 'class_score_aov'
class_score_aov <- S7::new_class(
  "class_score_aov",
  parent = class_score,
  properties = list(
    neg_log10 = S7::new_property(S7::class_logical, default = TRUE)
  )
)
```
:::

In addition to the properties inherited from the parent class, `class_score_aov` also includes:

- `neg_log10`: Represent the score as `-log10(p_value)`? It is `TRUE` or `FALSE`.

For the ANOVA F-test filter, users can represent the score using either

- p-value or 

- F-statistic. 

We demonstrate how to create these instances (objects) accordingly. 

We create `score_aov_pval` as an instance of the `class_score_aov` subclass by calling its constructor and specifying its properties:

::: {.cell layout-align="center"}

```{.r .cell-code}
# ANOVA p-value
score_aov_pval <-
  class_score_aov(
    outcome_type = c("numeric", "factor"),
    predictor_type = c("numeric", "factor"),
    case_weights = TRUE,
    range = c(0, Inf),
    inclusive = c(FALSE, FALSE),
    fallback_value = Inf,
    score_type = "aov_pval",
    direction = "maximize",
    deterministic = TRUE,
    tuning = FALSE,
    label = "ANOVA p-values"
  )
```
:::

Individual properties can be accessed via `object@`. For example: 

::: {.cell layout-align="center"}

```{.r .cell-code}
score_aov_pval@case_weights
#> [1] TRUE
score_aov_pval@fallback_value
#> [1] Inf
score_aov_pval@direction
#> [1] "maximize"
```
:::

`score_aov_fstat` is another instance of the `class_score_aov` subclass:  

::: {.cell layout-align="center"}

```{.r .cell-code}
# ANOVA F-statistic
score_aov_fstat <-
  class_score_aov(
    outcome_type = c("numeric", "factor"),
    predictor_type = c("numeric", "factor"),
    case_weights = TRUE,
    range = c(0, Inf),
    inclusive = c(FALSE, FALSE),
    fallback_value = Inf,
    score_type = "aov_fstat",
    direction = "maximize",
    deterministic = TRUE,
    tuning = FALSE,
    label = "ANOVA F-statistics"
  )
```
:::

## Fitting (or estimating) score

So far, we have discussed how to create a subclass and construct instances (objects) for the ANOVA F-test filter. 

`fit()` serves both as a generic and as method(s) used to fit (or estimate) score.

We define a generic named `fit()` that dispatches to the appropriate method based on the class of the object that is passed. We also define multiple methods named `fit()`. Each of these methods performs the actual fitting or score estimation for a specific class of object. 

The ANOVA F-test filter, for example: 

::: {.cell layout-align="center"}

```{.r .cell-code}
# Check the class of the object
class(score_aov_pval)
#> [1] "class_score_aov"     "filtro::class_score" "S7_object"
class(score_aov_fstat)
#> [1] "class_score_aov"     "filtro::class_score" "S7_object"
```
:::

Both objects belong to the (sub)class `class_score_aov`. Therefore, when `fit()` is called, the method for `class_score_aov` is dispatched, performing the actual fitting using the ANOVA F-test filter:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Method dispatch for objects of class `class_score_aov`
score_aov_pval |>
  fit(Sale_Price ~ ., data = ames)
score_aov_fstat |>
  fit(Sale_Price ~ ., data = ames)
```
:::

The correlation filter, for another example: 

::: {.cell layout-align="center"}

```{.r .cell-code}
# Check the class of the object
class(score_cor_pearson)
#> [1] "filtro::class_score_cor" "filtro::class_score"    
#> [3] "S7_object"
class(score_cor_spearman)
#> [1] "filtro::class_score_cor" "filtro::class_score"    
#> [3] "S7_object"
```
:::

Both objects belong to the (sub)class `class_score_cor`. Therefore, when `fit()` is called, the method for `class_score_cor` is dispatched, performing the actual fitting using the correlation filter:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Method dispatch for objects of class `class_score_cor`
score_cor_pearson |>
  fit(Sale_Price ~ ., data = ames)
score_cor_spearman |>
  fit(Sale_Price ~ ., data = ames)
```
:::

## Documenting S7 methods 

Documentation for S7 methods is still a work in progress, but here’s how we approached it. 

Instead of documenting each individual `fit()` method, we provide the details in the "Details" section and the "Estimating the scores" subsection of the documentation for the corresponding object. 

The code below opens the help page for the `fit()` generic: 

::: {.cell layout-align="center"}

```{.r .cell-code}
# Help page for `fit()` generic
?fit
```
:::

The code below opens the help pages for specific `fit()` methods: 

::: {.cell layout-align="center"}

```{.r .cell-code}
# Help page for `fit()` method along with the documentation for the specific object
?score_aov_pval
?score_aov_fstat
```
:::

::: {.cell layout-align="center"}

```{.r .cell-code}
# Help page for `fit()` method along with the documentation for the specific object
?score_cor_pearson
?score_cor_spearman
```
:::

## Accessing results after fitting

Once the method has been fitted via `fit()`, the data frame of results can be accessed via `object@results`. 

We use a subset of the Ames data set from the {modeldata} package for demonstration. The data set is used to predict housing sale price, and `Sale_Price` is the outcome and is numeric. 

::: {.cell layout-align="center"}

```{.r .cell-code}
library(modeldata)
ames_subset <- modeldata::ames |>
  # Use a subset of data for demonstration
  dplyr::select(
    Sale_Price,
    MS_SubClass,
    MS_Zoning,
    Lot_Frontage,
    Lot_Area,
    Street
  )
ames_subset <- ames_subset |>
  dplyr::mutate(Sale_Price = log10(Sale_Price))
```
:::

Next, we fit the score as we discuss before: 

::: {.cell layout-align="center"}

```{.r .cell-code}
# # Specify ANOVA p-value and fit score
# ames_aov_pval_res <-
#   score_aov_pval |>
#   fit(Sale_Price ~ ., data = ames_subset)
```
:::

::: {.cell layout-align="center"}

```{.r .cell-code}
# # Specify ANOVA F-statistic and fit score
# ames_aov_fstat_res <-
#   score_aov_fstat |>
#   fit(Sale_Price ~ ., data = ames_subset)
```
:::

Recall that individual properties of an object can be accessed using `object@`. Once the method has been fitted, the resulting data frame can be accessed via `object@results`:

::: {.cell layout-align="center"}

```{.r .cell-code}
# ames_aov_pval_res@results
```
:::

::: {.cell layout-align="center"}

```{.r .cell-code}
# ames_aov_fstat_res@results
```
:::

## Session information {#session-info}

::: {.cell layout-align="center"}

```
#> 
#> Attaching package: 'dplyr'
#> The following objects are masked from 'package:stats':
#> 
#>     filter, lag
#> The following objects are masked from 'package:base':
#> 
#>     intersect, setdiff, setequal, union
#> ─ Session info ─────────────────────────────────────────────────────
#>  version  R version 4.5.0 (2025-04-11)
#>  language (EN)
#>  date     2025-08-19
#>  pandoc   3.6.3
#>  quarto   1.7.32
#> 
#> ─ Packages ─────────────────────────────────────────────────────────
#>  package     version    date (UTC) source
#>  dplyr       1.1.4      2023-11-17 CRAN (R 4.5.0)
#>  filtro      0.1.0.9000 2025-08-15 Github (tidymodels/filtro@81c7d85)
#>  modeldata   1.4.0      2024-06-19 CRAN (R 4.5.0)
#>  purrr       1.0.4      2025-02-05 CRAN (R 4.5.0)
#>  rlang       1.1.6      2025-04-11 CRAN (R 4.5.0)
#>  tibble      3.2.1      2023-03-20 CRAN (R 4.5.0)
#> 
#> ────────────────────────────────────────────────────────────────────
```
:::

