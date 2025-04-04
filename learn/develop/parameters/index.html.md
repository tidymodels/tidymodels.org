---
title: "How to create a tuning parameter function"
categories:
  - developer tools
type: learn-subsection
weight: 4
description: | 
  Build functions to use in tuning both quantitative and qualitative parameters.
toc: true
toc-depth: 2
include-after-body: ../../../resources.html
---

## Introduction

To use code in this article,  you will need to install the following packages: dials and scales.

Some models and recipe steps contain parameters that dials does not know about. You can construct new quantitative and qualitative parameters using `new_quant_param()` or `new_qual_param()`, respectively. This article is a guide to creating new parameters.

## Quantitative parameters

As an example, let's consider the multivariate adaptive regression spline ([MARS](https://en.wikipedia.org/wiki/Multivariate_adaptive_regression_spline)) model, which creates nonlinear features from predictors and adds them to a linear regression models. The earth package is an excellent implementation of this method.

MARS creates an initial set of features and then prunes them back to an appropriate size. This can be done automatically by `earth::earth()` or the number of final terms can be set by the user. The parsnip function `mars()` has a parameter called `num_terms` that defines this.

What if we want to create a parameter for the number of *initial terms* included in the model. There is no argument in `parsnip::mars()` for this but we will make one now. The argument name in `earth::earth()` is `nk`, which is not very descriptive. Our parameter will be called `num_initial_terms`.

We use the `new_quant_param()` function since this is a numeric parameter. The main two arguments to a numeric parameter function are `range` and `trans`.

The `range` specifies the possible values of the parameter. For our example, a minimal value might be one or two. What is the upper limit? The default in the earth package is

::: {.cell layout-align="center"}

```{.r .cell-code}
min(200, max(20, 2 * ncol(x))) + 1
```
:::

where `x` is the predictor matrix. We often put in values that are either sensible defaults or are minimal enough to work for the majority of data sets. For now, let's specify an upper limit of 10 but this will be discussed more in the next section.

The other argument is `trans`, which represents a transformation that should be applied to the parameter values when working with them. For example, many regularization methods have a `penalty` parameter that tends to range between zero and some upper bound (let's say 1). The effect of going from a penalty value of 0.01 to 0.1 is much more impactful than going from 0.9 to 1.0. In such a case, it might make sense to work with this parameter in transformed units (such as the log, in this example). If new parameter values are generated at random, it helps if they are uniformly simulated in the transformed units and then converted back to the original units.

The `trans` parameter accepts a transformation object from the scales package. For example:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(scales)
lsf.str("package:scales", pattern = "_trans$")
#> asinh_trans : function ()  
#> asn_trans : function ()  
#> atanh_trans : function ()  
#> boxcox_trans : function (p, offset = 0)  
#> compose_trans : function (...)  
#> date_trans : function ()  
#> exp_trans : function (base = exp(1))  
#> hms_trans : function ()  
#> identity_trans : function ()  
#> log_trans : function (base = exp(1))  
#> log10_trans : function ()  
#> log1p_trans : function ()  
#> log2_trans : function ()  
#> logit_trans : function ()  
#> modulus_trans : function (p, offset = 1)  
#> probability_trans : function (distribution, ...)  
#> probit_trans : function ()  
#> pseudo_log_trans : function (sigma = 1, base = exp(1))  
#> reciprocal_trans : function ()  
#> reverse_trans : function ()  
#> sqrt_trans : function ()  
#> time_trans : function (tz = NULL)  
#> timespan_trans : function (unit = c("secs", "mins", "hours", "days", "weeks"))  
#> yj_trans : function (p)
scales::log10_trans()
#> Transformer: log-10 [1e-100, Inf]
```
:::

A value of `NULL` means that no transformation should be used.

A quantitative parameter function should have these two arguments and, in the function body, a call `new_quant_param()`. There are a few arguments to this function:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(tidymodels)
args(new_quant_param)
#> function (type = c("double", "integer"), range = NULL, inclusive = NULL, 
#>     default = deprecated(), trans = NULL, values = NULL, label = NULL, 
#>     finalize = NULL, ..., call = caller_env()) 
#> NULL
```
:::

-   Possible types are double precision and integers. The value of `type` should agree with the values of `range` in the function definition.

-   It's OK for our tuning to include the minimum or maximum, so we'll use `c(TRUE, TRUE)` for `inclusive`. If the value cannot include one end of the range, set one or both of these values to `FALSE`.

-   The `label` should be a named character string where the name is the parameter name and the value represents what will be printed automatically.

-   `finalize` is an argument that can set parts of the range. This is discussed more below.

Here's an example of a basic quantitative parameter object:

::: {.cell layout-align="center"}

```{.r .cell-code}
num_initial_terms <- function(range = c(1L, 10L), trans = NULL) {
  new_quant_param(
    type = "integer",
    range = range,
    inclusive = c(TRUE, TRUE),
    trans = trans,
    label = c(num_initial_terms = "# Initial MARS Terms"),
    finalize = NULL
  )
}

num_initial_terms()
#> # Initial MARS Terms (quantitative)
#> Range: [1, 10]

# Sample from the parameter:
set.seed(4832856)
num_initial_terms() %>% value_sample(5)
#> [1]  6  4  9 10  4
```
:::

### Finalizing parameters

It might be the case that the range of the parameter is unknown. For example, parameters that are related to the number of columns in a data set cannot be exactly specified in the absence of data. In those cases, a placeholder of `unknown()` can be added. This will force the user to "finalize" the parameter object for their particular data set. Let's redefine our function with an `unknown()` value:

::: {.cell layout-align="center"}

```{.r .cell-code}
num_initial_terms <- function(range = c(1L, unknown()), trans = NULL) {
  new_quant_param(
    type = "integer",
    range = range,
    inclusive = c(TRUE, TRUE),
    trans = trans,
    label = c(num_initial_terms = "# Initial MARS Terms"),
    finalize = NULL
  )
}
num_initial_terms()

# Can we sample? 
num_initial_terms() %>% value_sample(5)
```
:::

The `finalize` argument of `num_initial_terms()` can take a function that uses data to set the range. For example, the package already includes a few functions for finalization:

::: {.cell layout-align="center"}

```{.r .cell-code}
lsf.str("package:dials", pattern = "^get_")
#> get_batch_sizes : function (object, x, frac = c(1/10, 1/3), ...)  
#> get_log_p : function (object, x, ...)  
#> get_n : function (object, x, log_vals = FALSE, ...)  
#> get_n_frac : function (object, x, log_vals = FALSE, frac = 1/3, ...)  
#> get_n_frac_range : function (object, x, log_vals = FALSE, frac = c(1/10, 5/10), ...)  
#> get_p : function (object, x, log_vals = FALSE, ...)  
#> get_rbf_range : function (object, x, seed = sample.int(10^5, 1), ...)
```
:::

These functions generally take a data frame of predictors (in an argument called `x`) and add the range of the parameter object. Using the formula in the earth package, we might use:

::: {.cell layout-align="center"}

```{.r .cell-code}
get_initial_mars_terms <- function(object, x) {
  upper_bound <- min(200, max(20, 2 * ncol(x))) + 1
  upper_bound <- as.integer(upper_bound)
  bounds <- range_get(object)
  bounds$upper <- upper_bound
  range_set(object, bounds)
}

# Use the mtcars are the finalize the upper bound: 
num_initial_terms() %>% get_initial_mars_terms(x = mtcars[, -1])
#> # Initial MARS Terms (quantitative)
#> Range: [1, 21]
```
:::

Once we add this function to the object, the general `finalize()` method can be used:

::: {.cell layout-align="center"}

```{.r .cell-code}
num_initial_terms <- function(range = c(1L, unknown()), trans = NULL) {
  new_quant_param(
    type = "integer",
    range = range,
    inclusive = c(TRUE, TRUE),
    trans = trans,
    label = c(num_initial_terms = "# Initial MARS Terms"),
    finalize = get_initial_mars_terms
  )
}

num_initial_terms() %>% finalize(x = mtcars[, -1])
#> # Initial MARS Terms (quantitative)
#> Range: [1, 21]
```
:::

## Qualitative parameters

Now let's look at an example of a qualitative parameter. If a model includes a data aggregation step, we want to allow users to tune how our parameters are aggregated. For example, in embedding methods, possible values might be `min`, `max`, `mean`, `sum`, or to not aggregate at all ("none"). Since these cannot be put on a numeric scale, they are possible values of a qualitative parameter. We'll take "character" input (not "logical"), and we must specify the allowed values. By default we won't aggregate.

::: {.cell layout-align="center"}

```{.r .cell-code}
aggregation <- function(values = c("none", "min", "max", "mean", "sum")) {
  new_qual_param(
    type = "character",
    values = values,
    # By default, the first value is selected as default. We'll specify that to
    # make it clear.
    default = "none",
    label = c(aggregation = "Aggregation Method")
  )
}
```
:::

Within the dials package, the convention is to have the values contained in a separate vector whose name starts with `values_`. For example:

::: {.cell layout-align="center"}

```{.r .cell-code}
values_aggregation <- c("none", "min", "max", "mean", "sum")
aggregation <- function(values = values_aggregation) {
  new_qual_param(
    type = "character",
    values = values,
    # By default, the first value is selected as default. We'll specify that to
    # make it clear.
    label = c(aggregation = "Aggregation Method")
  )
}
```
:::

This step may not make sense if you are using the function in a script and not keeping it within a package.

We can use our `aggregation` parameters with dials functions.

::: {.cell layout-align="center"}

```{.r .cell-code}
aggregation()
#> Aggregation Method (qualitative)
#> 5 possible values include:
#> 'none', 'min', 'max', 'mean', and 'sum'
aggregation() %>% value_sample(3)
#> [1] "min"  "sum"  "mean"
```
:::

## Session information {#session-info}

::: {.cell layout-align="center"}

```
#> ─ Session info ─────────────────────────────────────────────────────
#>  version  R version 4.4.2 (2024-10-31)
#>  language (EN)
#>  date     2025-03-24
#>  pandoc   3.6.1
#>  quarto   1.6.42
#> 
#> ─ Packages ─────────────────────────────────────────────────────────
#>  package      version date (UTC) source
#>  broom        1.0.7   2024-09-26 CRAN (R 4.4.1)
#>  dials        1.4.0   2025-02-13 CRAN (R 4.4.2)
#>  dplyr        1.1.4   2023-11-17 CRAN (R 4.4.0)
#>  ggplot2      3.5.1   2024-04-23 CRAN (R 4.4.0)
#>  infer        1.0.7   2024-03-25 CRAN (R 4.4.0)
#>  parsnip      1.3.1   2025-03-12 CRAN (R 4.4.1)
#>  purrr        1.0.4   2025-02-05 CRAN (R 4.4.1)
#>  recipes      1.2.0   2025-03-17 CRAN (R 4.4.1)
#>  rlang        1.1.5   2025-01-17 CRAN (R 4.4.2)
#>  rsample      1.2.1   2024-03-25 CRAN (R 4.4.0)
#>  scales       1.3.0   2023-11-28 CRAN (R 4.4.0)
#>  tibble       3.2.1   2023-03-20 CRAN (R 4.4.0)
#>  tidymodels   1.3.0   2025-02-21 CRAN (R 4.4.1)
#>  tune         1.3.0   2025-02-21 CRAN (R 4.4.1)
#>  workflows    1.2.0   2025-02-19 CRAN (R 4.4.1)
#>  yardstick    1.3.2   2025-01-22 CRAN (R 4.4.1)
#> 
#> ────────────────────────────────────────────────────────────────────
```
:::
