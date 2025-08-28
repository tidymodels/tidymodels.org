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

filtro is tidy tools to apply filter-based supervised feature selection methods. It provides functions to rank and select a specified proportion or a fixed number of features using built-in methods and the desirability function. 

Currently, there are 6 filters in filtro and many existing score objects. A list of existing scoring objects [can be found here](https://filtro.tidymodels.org/articles/filtro.html#available-score-objects-and-filter-methods). 

However, you might need to define your own scoring objects. 

This article serves as a guide to creating new scoring objects and computing feature scores before performing ranking and selection. 

The general procedure is to:

1. Construct a parent scoring object using `class_score()`, specifying fixed properties. 

2. Construct a custom scoring object using `class_score_*()`, defining additional properties.

3. Define the scoring method in `fit()` to compute feature score. `fit()` refers to the custom scoring object from Step 2 to determine which method to dispatch.

As an example, we will walk through the steps to create an ANOVA F-test filter.

## Scoring object

All the custom scoring objects share the same parent class named `class_score`. Therefore, we start by constructing a new scoring object using `class_score()`. 

These are the fixed properties (attributes) for this object:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(filtro)
args(class_score)
#> function (outcome_type = c("numeric", "factor"), predictor_type = c("numeric", 
#> "factor"), case_weights = logical(0), range = integer(0), inclusive = logical(0), 
#>     fallback_value = integer(0), score_type = character(0), transform_fn = function() NULL, 
#>     direction = character(0), deterministic = logical(0), tuning = logical(0), 
#>     calculating_fn = function() NULL, label = character(0), packages = character(0), 
#>     results = data.frame()) 
#> NULL
```
:::

For example: 

-   `case_weights`: Does the method accpet case weights? It is `TRUE` or `FALSE`.

-   `fallback_value`: What is a value that can be used for the statistic so that it will never be eliminated? For example, `0`, `Inf`.

-   `direction`: What direction of values indicates the most important values? For example,  `maximum`, `minimize`.

-   `results`: A slot for the results once the method is fitted. Initially, this is an empty data frame.

For details on the remaining properties, please refer to the package documentation.

## Custom scoring object

Next, we demonstrate how to create a custom scoring object. 

As an example, let’s create a custom scoring object for ANOVA F-test named `class_score_aov`. This filter computes feature score using analysis of variance (ANOVA) hypothesis tests, powered by `lm()`. 

`class_score_aov` is a subclass of `class_score`. It inherits all fixed properties from the parent class, while allowing additional implementation-specific properties to be added in the subclass. For example:

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

In addition to the properties inherited from the parent class (discussed in the previous section), `class_score_aov` also includes: 

- `neg_log10`: Represent the score as `-log10(p_value)`? It is `TRUE` or `FALSE`.

For the ANOVA F-test filter, users can represent the score using either

- p-value or 

- F-statistic. 

We demonstrate how to create these instances (objects) accordingly. 

`score_aov_pval` is created as an instance of the `class_score_aov` subclass by calling its constructor and specifying its properties:

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

## Fitting (or estimating) feature score

So far, we have covered how to construct a parent class, create a custom subclass, and instantiate objects for the ANOVA F-test filter. 

We now discuss the dual role of `fit()`: it functions both as a generic and as the methods used to fit (or estimate) feature score. 

1. The generic `fit()` is re-exported from generics. It inspects the class of the object passed and dispatches to the appropriate method. 

2. We also define multiple methods named `fit()`. Each method `fit()` performs the actual fitting or score estimation for a specific class of object. 

In other words, when `fit()` is called, the generic refers to the custom scoring object (rather than the parent scoring object) to determine which method to dispatch. The actual scoring computation is performed within the dispatched method. 

The ANOVA F-test filter, for example: 

::: {.cell layout-align="center"}

```{.r .cell-code}
# User-level example: Check the class of the object
class(score_aov_pval)
#> [1] "filtro::class_score_aov" "filtro::class_score"    
#> [3] "S7_object"
class(score_aov_fstat)
#> [1] "filtro::class_score_aov" "filtro::class_score"    
#> [3] "S7_object"
```
:::

Both instances (objects) belong to the custom scoring object `class_score_aov`. Therefore, when `fit()` is called, the method for `class_score_aov` is dispatched, performing the actual fitting using the ANOVA F-test:

::: {.cell layout-align="center"}

```{.r .cell-code}
# User-level example: Method dispatch for objects of class `class_score_aov`
score_aov_pval |>
  fit(Sale_Price ~ ., data = ames)
score_aov_fstat |>
  fit(Sale_Price ~ ., data = ames)
```
:::

## Defining S7 methods 

To use the `fit()` method, we need to define S7 method that implements the scoring logic: 

::: {.cell layout-align="center"}

```{.r .cell-code}
# Define the scoring method for `class_score_aov`
#' @export
S7::method(fit, class_score_aov) <- function(
  object,
  formula,
  data,
  case_weights = NULL,
  ...
) {
  # TODO finish the rest of the function

  object@results <- res
  object
}
```
:::

## Documenting S7 methods 

Documentation for S7 methods is still a work in progress, and it seems no one currently knows the right approach. Here’s how we tackle it: 

We re-export the `fit()` generic from generics. Instead of documenting each `fit()` method, we provide the details in the "Details" section and the "Estimating the scores" subsection of the documentation for the corresponding object. 

The code below opens the help page for the `fit()` generic: 

::: {.cell layout-align="center"}

```{.r .cell-code}
# User-level example: Help page for `fit()` generic
?fit
```
:::

The code below opens the help page for specific `fit()` method: 

::: {.cell layout-align="center"}

```{.r .cell-code}
# User-level example: Help page for `fit()` method along with the documentation for the specific object
?score_aov_pval
?score_aov_fstat
```
:::

## Accessing results after fitting

Once the method has been fitted via `fit()`, the data frame of results can be accessed via `object@results`. 

We use a subset of the Ames data set from the {modeldata} package for demonstration. The goal is to predict housing sale price. `Sale_Price` is the outcome and is numeric. 

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
# Specify ANOVA p-value and fit score
ames_aov_pval_res <-
  score_aov_pval |>
  fit(Sale_Price ~ ., data = ames_subset)
```
:::

::: {.cell layout-align="center"}

```{.r .cell-code}
# Specify ANOVA F-statistic and fit score
ames_aov_fstat_res <-
  score_aov_fstat |>
  fit(Sale_Price ~ ., data = ames_subset)
```
:::

Recall that individual properties of an object can be accessed using `object@`. Once the method has been fitted, the resulting data frame can be accessed via `object@results`:

::: {.cell layout-align="center"}

```{.r .cell-code}
ames_aov_pval_res@results
#> # A tibble: 5 × 4
#>   name      score outcome    predictor   
#>   <chr>     <dbl> <chr>      <chr>       
#> 1 aov_pval 237.   Sale_Price MS_SubClass 
#> 2 aov_pval 130.   Sale_Price MS_Zoning   
#> 3 aov_pval  NA    Sale_Price Lot_Frontage
#> 4 aov_pval  NA    Sale_Price Lot_Area    
#> 5 aov_pval   5.75 Sale_Price Street
```
:::

::: {.cell layout-align="center"}

```{.r .cell-code}
ames_aov_fstat_res@results
#> # A tibble: 5 × 4
#>   name      score outcome    predictor   
#>   <chr>     <dbl> <chr>      <chr>       
#> 1 aov_fstat  94.5 Sale_Price MS_SubClass 
#> 2 aov_fstat 115.  Sale_Price MS_Zoning   
#> 3 aov_fstat  NA   Sale_Price Lot_Frontage
#> 4 aov_fstat  NA   Sale_Price Lot_Area    
#> 5 aov_fstat  22.9 Sale_Price Street
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
#>  date     2025-08-28
#>  pandoc   3.6.3
#>  quarto   1.7.32
#> 
#> ─ Packages ─────────────────────────────────────────────────────────
#>  package     version    date (UTC) source
#>  dplyr       1.1.4      2023-11-17 CRAN (R 4.5.0)
#>  filtro      0.1.0.9000 2025-08-26 Github (tidymodels/filtro@f8ffd50)
#>  modeldata   1.4.0      2024-06-19 CRAN (R 4.5.0)
#>  purrr       1.0.4      2025-02-05 CRAN (R 4.5.0)
#>  rlang       1.1.6      2025-04-11 CRAN (R 4.5.0)
#>  tibble      3.2.1      2023-03-20 CRAN (R 4.5.0)
#> 
#> ────────────────────────────────────────────────────────────────────
```
:::

