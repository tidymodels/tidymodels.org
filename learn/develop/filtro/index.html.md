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

Currently, there are 6 filters in filtro and many existing score objects. A list of existing scoring objects [can be found here](https://filtro.tidymodels.org/articles/filtro.html#available-score-objects-and-filter-methods). However, you might need to define your own scoring objects. This article serves as a guide to creating new scoring objects and computing feature scores before performing ranking and selection. 

For reference, filtro is tidy tools to apply filter-based supervised feature selection methods. It provides functions to rank and select a specified proportion or a fixed number of features using built-in methods and the desirability function. 

Regarding scoring objects: 

There is a parent class `class_score`, which defines the fixed properties that are shared across all subclasses. The parent class is already implemented, and serves as the infrastructure we build on when we make our own scoring class object.

Therefore, the general procedure is to:

1. Create a subclass `class_score_*` that inherts from `class_score`. This subclass introduces additional, method- or score-specific properties, as opposed to the general characteristics already defined in the parent class. 

2. Implement the scoring method in `fit()`, which computes feature score. The `fit()` generic refers to the subclass from step 1 to use the appropriate `fit()` method. 

The hierarchy can be visualized as:

```
class_score
└─> class_score_* 
 └─> fit()
```

Additionally, we provide guidance on documenting an S7 method.

## Parent class (General scoring object)

All the subclasses (custom scoring objects) share the same parent class named `class_score`. The parent class is already implemented, and we need this to build our own scoring class object: 

::: {.cell layout-align="center"}

```{.r .cell-code}
# Call the parent class
library(filtro) 
class_score
```
:::

These are the fixed properties (attributes) for this object:

::: {.cell layout-align="center"}

```{.r .cell-code}
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

-   `fallback_value`: What is a value that can be used for the statistic so that it will never be eliminated? For example, `0` or `Inf`.

-   `direction`: What direction of values indicates the most important values? For example, `minimize` or `maximum`.

-   `results`: A slot for the results once the method is fitted. Initially, this is an empty data frame.

For details on its constructor and its remaining properties, please refer to the package documentation or [check it out here](https://github.com/tidymodels/filtro/blob/main/R/class_score.R). 

## Subclass (Custom scoring object)

All custom scoring objects implemented in filtro are subclasses of the parent class `class_score`, meaning that they all inherit the parent class's fixed properties. When creating a new scoring object, we do so by defining another subclass of this parent class.

```
class_score
└─> class_score_aov (example shown)
└─> class_score_cor
└─> ... 
```

As an example, we demonstrate how to create a custom scoring object for ANOVA F-test named `class_score_aov`. 

For reference, the ANOVA F-test filter computes feature score using analysis of variance (ANOVA) hypothesis tests, powered by `lm()`. The `lm()` function fits a linear model and returns a summary containing the F-statistic and p-value, which can be used to evaluate feature importance. 

By setting `parent = class_score`, the subclass `class_score_aov` inherits all of the fixed properties from the parent class. Additional, implementation-specific properties can be added using the `properties =` argument. For example:

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

In addition to the properties inherited from the parent class (discussed in the previous section), `class_score_aov` also includes the following property:

- `neg_log10`: Represent the score as `-log10(p_value)`? It is `TRUE` or `FALSE`.

For the ANOVA F-test filter, users can represent the score using either the 

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

Once instantiated, individual properties can be accessed via `object@`. For example: 

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

Note that by default, the returned p-value is transformed to `-log10(p_value)`, which means larger values correspond to more important predictors. This is why the fallback value is set to `Inf` and the direction is set to `"maximize"`. 

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

The F-statistic is not transformed, nor does it provide an option for transformation. Nevertheless, it also uses a fallback value of `Inf` with the direction set to `"maximize"`, since larger F-statistic values indicate more important predictors. 

## Fitting (or estimating) feature score

So far, we have covered how to construct a parent class, create a custom subclass, and instantiate objects for the ANOVA F-test filter. 

We now discuss the dual role of `fit()`: it functions both as a *generic* and as the *methods* used to fit (or estimate) feature score. 

1. The `fit()` generic is re-exported from generics. It inspects the class of the object passed and dispatches to the appropriate method. 

2. We also define multiple methods named `fit()`. Each `fit()` method performs the actual fitting or score estimation for a specific class of object. 

In other words, when `fit()` is called, the generic refers to the custom scoring object `class_score_*` to determine which method to dispatch. The actual scoring computation is performed within the dispatched method. 

```
class_score
└─> class_score_aov (example shown)
 └─> fit()
└─> class_score_cor
 └─> fit()
└─> ... 
```

Let’s use the ANOVA F-test filter again as an example: 

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

For users to use the `fit()` method described above, we need to define a S7 method that implements the scoring logic. 

The following code defines the `fit()` method specifically for the `class_score_aov` subclass, specifying how feature score should be computed using ANOVA F-test:

::: {.cell layout-align="center"}

```{.r .cell-code}
# Define the scoring method for `class_score_aov`
S7::method(fit, class_score_aov) <- function(
  object,
  formula,
  data,
  case_weights = NULL,
  ...
) {
  # This is where you add the rest of the code for this implementation 

  object@results <- res
  object
}
```
:::

We would want to do something similar to define a S7 method for other `class_score_*` subclass. 

## Documenting S7 methods 

Documentation for S7 methods is still a work in progress, but our current best approach is as follows: 

- We re-export the `fit()` generic from generics. 

- Instead of documenting each `fit()` method, we document it in the "Details" section and the "Estimating the scores" subsection of the documentation for the corresponding object (instance) `score_*`. 

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

For users to access the help page using `?` as described above, the `fit()` method needs to be exported using `#' @export`, but it is not documented directly.

::: {.cell layout-align="center"}

```{.r .cell-code}
#' @export
S7::method(fit, class_score_aov) <- function(
  object,
  ...
) {
  # Include the rest of the function here

  object@results <- res
  object
}
```
:::

Instead, documentation is provided in the "Details" section and the "Estimating the scores" subsection of the documentation for the `score_aov_pval` object.  

::: {.cell layout-align="center"}

```{.r .cell-code}
#' Scoring via analysis of variance hypothesis tests
#'
#' @description
#' 
#' @name score_aov_pval
#' @family class score metrics
#'
#' @details
#'
#' These objects are used when either:
#'
#' ...
#'
#' ## Estimating the scores
#'
#' In \pkg{filtro}, the `score_*` objects define a scoring method (e.g., data
#' input requirements, package dependencies, etc). To compute the scores for
#' a specific data set, the `fit()` method is used. The main arguments for
#' these functions are:
#'
#'   \describe{
#'     \item{`object`}{A score class object (e.g., `score_aov_pval`).}
#'     \item{`formula`}{A standard R formula with a single outcome on the right-hand side and one or more predictors (or `.`) on the left-hand side. The data are processed via [stats::model.frame()]}
#'     \item{`data`}{A data frame containing the relevant columns defined by the formula.}
#'     \item{`...`}{Further arguments passed to or from other methods.}
#'     \item{`case_weights`}{A quantitative vector of case weights that is the same length as the number of rows in `data`. The default of `NULL` indicates that there are no case weights.}
#'   }
#'
#' ...
#' 
#' @export
score_aov_pval <-
  class_score_aov(
    outcome_type = c("numeric", "factor"),
    predictor_type = c("numeric", "factor"),
    case_weights = TRUE,
    range = c(0, Inf),
    inclusive = c(FALSE, FALSE),
    fallback_value = Inf,
    score_type = "aov_pval",
    transform_fn = function(x) x,
    direction = "maximize",
    deterministic = TRUE,
    tuning = FALSE,
    label = "ANOVA p-values"
  )
```
:::

We can have the `score_aov_fstat` object share the same help page as `score_aov_pval` by using `#' @name`. This avoids repeated documentation for similar or related objects.

::: {.cell layout-align="center"}

```{.r .cell-code}
#' @name score_aov_pval
#' @export
score_aov_fstat <-
  class_score_aov(
    outcome_type = c("numeric", "factor"),
    predictor_type = c("numeric", "factor"),
    case_weights = TRUE,
    range = c(0, Inf),
    inclusive = c(FALSE, FALSE),
    fallback_value = Inf,
    score_type = "aov_fstat",
    transform_fn = function(x) x,
    direction = "maximize",
    deterministic = TRUE,
    tuning = FALSE,
    label = "ANOVA F-statistics"
  )
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
#>  version  R version 4.5.1 (2025-06-13)
#>  language (EN)
#>  date     2025-10-07
#>  pandoc   3.6.3
#>  quarto   1.8.25
#> 
#> ─ Packages ─────────────────────────────────────────────────────────
#>  package     version date (UTC) source
#>  dplyr       1.1.4   2023-11-17 CRAN (R 4.5.0)
#>  filtro      0.2.0   2025-08-26 CRAN (R 4.5.0)
#>  modeldata   1.5.1   2025-08-22 CRAN (R 4.5.0)
#>  purrr       1.1.0   2025-07-10 CRAN (R 4.5.0)
#>  rlang       1.1.6   2025-04-11 CRAN (R 4.5.0)
#>  tibble      3.3.0   2025-06-08 CRAN (R 4.5.0)
#> 
#> ────────────────────────────────────────────────────────────────────
```
:::

