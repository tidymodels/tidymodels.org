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

The `class_score` is the parent class of all subclasses related to the scoring method. There are a few properties to this object:

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

-   `outcome_type`: What types of outcome can the method handle?

-   `predictor_type`: What types of predictor can the method handle?

-   `case_weights`: Does the method accpet case weights? 

-   `range`: Are there known ranges for the statistic?

-   `inclusive`: Are these ranges inclusive at the bounds?

-   `fallback_value`: What is a value that can be used for the statistic so that it will never be eliminated?

-   `score_type`: What is the column name that will be used for the statistic values?

-   (Not used) `sorts`: How should the values be sorted (from most- to least-important)?

-   `direction`: What direction of values indicates the most important values?

-   `deterministic`: Does the fitting process use random numbers?

-   `tuning`: Does the method have tuning parameters?

-   `calculating_fn`: What function is used to estimate the values from data?

-   `label`: What label to use when printing?

-   `packages`: What packages, if any, are required to train the method?

-   `results`: A slot for the results once the method is fitted.

## Scoring object specific to the scoring method

As an example, let’s consider the ANOVA F-test filter. 

`class_score_aov` is a subclass of `class_score`. Because it inherits from the `class_score` parent class, all of the parent's properties are also inherited. The subclass can also include additional properties specific to their implementation. For example: 

::: {.cell layout-align="center"}

```{.r .cell-code}
class_score_aov <- S7::new_class(
  "class_score_aov",
  parent = class_score,
  properties = list(
    # Represent the score as -log10(p_value)?
    neg_log10 = S7::new_property(S7::class_logical, default = TRUE)
  )
)
```
:::

For this filter, users can use either p-value or the F-statistic. We will demonstrate how to create these instances (or objects).

`score_aov_pval` is an instance (or object) of the `class_score_aov` subclass, created using its constructor function: 

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

Individual properties can be accessed via `object@`. For examples: 

::: {.cell layout-align="center"}

```{.r .cell-code}
score_aov_pval@case_weights
#> [1] TRUE
score_aov_pval@range
#> [1]   0 Inf
score_aov_pval@fallback_value
#> [1] Inf
```
:::

`score_aov_fstat` is another instance (or object) of the `class_score_aov` subclass:  

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

## Accessing Results After Fitting

Once the method is fitted via `fit()`, the data frame of results can be accessed via `object@results`. For examples: 

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

::: {.cell layout-align="center"}

```{.r .cell-code}
# # Specify ANOVA p-value and fit score
# ames_aov_pval_res <-
#   score_aov_pval |>
#   fit(Sale_Price ~ ., data = ames_subset)
# ames_aov_pval_res@results
```
:::

::: {.cell layout-align="center"}

```{.r .cell-code}
# # Specify ANOVA F-statistic and fit score
# ames_aov_fstat_res <-
#   score_aov_fstat |>
#   fit(Sale_Price ~ ., data = ames_subset)
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
#>  date     2025-08-18
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

