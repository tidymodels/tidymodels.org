---
title: "Custom performance metrics"
categories:
  - developer tools
type: learn-subsection
weight: 3
description: | 
  Create a new performance metric and integrate it with yardstick functions.
toc: true
toc-depth: 2
include-after-body: ../../../resources.html
---

## Introduction

To use code in this article,  you will need to install the following packages: rlang and tidymodels.

The [yardstick](https://yardstick.tidymodels.org/) package already includes a large number of metrics, but there's obviously a chance that you might have a custom metric that hasn't been implemented yet. In that case, you can use a few of the tools yardstick exposes to create custom metrics.

Why create custom metrics? With the infrastructure yardstick provides, you get:

-   Standardization between your metric and other preexisting metrics
-   Automatic error handling for types and lengths
-   Automatic selection of binary / multiclass metric implementations
-   Support for `NA` handling
-   Support for grouped data frames
-   Support for use alongside other metrics in `metric_set()`

The implementation for metrics differ slightly depending on whether you are implementing a numeric, class, or class probability metric. Examples for numeric and classification metrics are given below. We would encourage you to look into the implementation of `roc_auc()` after reading this vignette if you want to work on a class probability metric.

## Numeric example: MSE

Mean squared error (sometimes MSE or from here on, `mse()`) is a numeric metric that measures the average of the squared errors. Numeric metrics are generally the simplest to create with yardstick, as they do not have multiclass implementations. The formula for `mse()` is:

$$ MSE = \frac{1}{N} \sum_{i=1}^{N} (truth_i - estimate_i) ^ 2 = mean( (truth - estimate) ^ 2) $$

All metrics should have a data frame version, and a vector version. The data frame version here will be named `mse()`, and the vector version will be `mse_vec()`.

### Vector implementation

To start, create the vector version. Generally, all metrics have the same arguments unless the metric requires an extra parameter (such as `beta` in `f_meas()`). To create the vector function, you need to do two things:

1)  Create an internal implementation function, `mse_impl()`.
2)  Use that implementation function with `check_class_metric()`/`check_numeric_metric()`/`check_prob_metric()`, `yardstick_remove_missing()`, and `and yardstick_any_missing()`.

Below, `mse_impl()` contains the actual implementation of the metric, and takes `truth` and `estimate` as arguments along with any metric specific arguments. Optionally `case_weights` if the calculations supports it.

The yardstick function `check_numeric_metric()` takes `truth`, `estimate` and `case_weights`, and validates that they are the right type, and are the same length.

The `yardstick_remove_missing()` and `yardstick_any_missing()` yardstick functions are used to handle missing values in a consistent way, similarly to how the other metrics handle them. The code below is typically copy pasted from function to function, but certain types of metrics might want to deviate from this pattern.

You are required to supply a `case_weights` argument to `mse_vec()` for the functions to work with yardstick. If your metric in question doesn't support case weights, you can error if they are passed, or simply ignore it.

::: {.cell layout-align="center"}

```{.r .cell-code}
library(tidymodels)

mse_impl <- function(truth, estimate, case_weights = NULL) {
  mean((truth - estimate) ^ 2)
}

mse_vec <- function(truth, estimate, na_rm = TRUE, case_weights = NULL, ...) {
  check_numeric_metric(truth, estimate, case_weights)

  if (na_rm) {
    result <- yardstick_remove_missing(truth, estimate, case_weights)

    truth <- result$truth
    estimate <- result$estimate
    case_weights <- result$case_weights
  } else if (yardstick_any_missing(truth, estimate, case_weights)) {
    return(NA_real_)
  }

  mse_impl(truth, estimate, case_weights = case_weights)
}
```
:::

At this point, you've created the vector version of the mean squared error metric.

::: {.cell layout-align="center"}

```{.r .cell-code}
data("solubility_test")

mse_vec(
  truth = solubility_test$solubility, 
  estimate = solubility_test$prediction
)
#> [1] 0.5214438
```
:::

Intelligent error handling is immediately available.

::: {.cell layout-align="center"}

```{.r .cell-code}
mse_vec(truth = "apple", estimate = 1)
#> Error in `mse_vec()`:
#> ! `truth` should be a numeric vector, not a string.

mse_vec(truth = 1, estimate = factor("xyz"))
#> Error in `mse_vec()`:
#> ! `estimate` should be a numeric vector, not a <factor>
#>   object.
```
:::

`NA` values are removed if `na_rm = TRUE` (the default). If `na_rm = FALSE` and any `NA` values are detected, then the metric automatically returns `NA`.

::: {.cell layout-align="center"}

```{.r .cell-code}
# NA values removed
mse_vec(truth = c(NA, .5, .4), estimate = c(1, .6, .5))
#> [1] 0.01

# NA returned
mse_vec(truth = c(NA, .5, .4), estimate = c(1, .6, .5), na_rm = FALSE)
#> [1] NA
```
:::

### Data frame implementation

The data frame version of the metric should be fairly simple. It is a generic function with a `data.frame` method that calls the yardstick helper, `numeric_metric_summarizer()`, and passes along the `mse_vec()` function to it along with versions of `truth` and `estimate` that have been wrapped in `rlang::enquo()` and then unquoted with `!!` so that non-standard evaluation can be supported.

::: {.cell layout-align="center"}

```{.r .cell-code}
library(rlang)

mse <- function(data, ...) {
  UseMethod("mse")
}

mse <- new_numeric_metric(mse, direction = "minimize")

mse.data.frame <- function(data, truth, estimate, na_rm = TRUE, case_weights = NULL, ...) {

  numeric_metric_summarizer(
    name = "mse",
    fn = mse_vec,
    data = data,
    truth = !!enquo(truth),
    estimate = !!enquo(estimate),
    na_rm = na_rm,
    case_weights = !!enquo(case_weights)
  )
}
```
:::

And that's it. The yardstick package handles the rest.

::: {.cell layout-align="center"}

```{.r .cell-code}
mse(solubility_test, truth = solubility, estimate = prediction)

# Error handling
mse(solubility_test, truth = solubility, estimate = factor("xyz"))
```
:::

Let's test it out on a grouped data frame.

::: {.cell layout-align="center"}

```{.r .cell-code}
library(dplyr)

set.seed(1234)
size <- 100
times <- 10

# create 10 resamples
solubility_resampled <- bind_rows(
  replicate(
    n = times,
    expr = sample_n(solubility_test, size, replace = TRUE),
    simplify = FALSE
  ),
  .id = "resample"
)

solubility_resampled %>%
  group_by(resample) %>%
  mse(solubility, prediction)
#> # A tibble: 10 × 4
#>    resample .metric .estimator .estimate
#>    <chr>    <chr>   <chr>          <dbl>
#>  1 1        mse     standard       0.512
#>  2 10       mse     standard       0.454
#>  3 2        mse     standard       0.513
#>  4 3        mse     standard       0.414
#>  5 4        mse     standard       0.543
#>  6 5        mse     standard       0.456
#>  7 6        mse     standard       0.652
#>  8 7        mse     standard       0.642
#>  9 8        mse     standard       0.404
#> 10 9        mse     standard       0.479
```
:::

## Class example: miss rate

Miss rate is another name for the false negative rate, and is a classification metric in the same family as `sens()` and `spec()`. It follows the formula:

$$ miss\_rate = \frac{FN}{FN + TP} $$

This metric, like other classification metrics, is more easily computed when expressed as a confusion matrix. As you will see in the example, you can achieve this with a call to `base::table(estimate, truth)` which correctly puts the "correct" result in the columns of the confusion matrix.

Classification metrics are more complicated than numeric ones because you have to think about extensions to the multiclass case. For now, let's start with the binary case.

### Vector implementation

The vector implementation for classification metrics initially has a very similar setup as the numeric metrics. It used `check_class_metric()` instead of `check_numeric_metric()`. It has an additional argument, `estimator` that determines the type of estimator to use (binary or some kind of multiclass implementation or averaging). This argument is auto-selected for the user, so default it to  `NULL`. Additionally, pass it along to `check_class_metric()` so that it can check the provided `estimator` against the classes of `truth` and `estimate` to see if they are allowed.

::: {.cell layout-align="center"}

```{.r .cell-code}
# Logic for `event_level`
event_col <- function(xtab, event_level) {
  if (identical(event_level, "first")) {
    colnames(xtab)[[1]]
  } else {
    colnames(xtab)[[2]]
  }
}

miss_rate_impl <- function(truth, estimate, event_level) {
  # Create 
  xtab <- table(estimate, truth)
  col <- event_col(xtab, event_level)
  col2 <- setdiff(colnames(xtab), col)

  tp <- xtab[col, col]
  fn <- xtab[col2, col]

  fn / (fn + tp)
}

miss_rate_vec <- function(truth,
                          estimate,
                          estimator = NULL,
                          na_rm = TRUE,
                          case_weights = NULL,
                          event_level = "first",
                          ...) {
  estimator <- finalize_estimator(truth, estimator)

  check_class_metric(truth, estimate, case_weights, estimator)
  
    if (na_rm) {
    result <- yardstick_remove_missing(truth, estimate, case_weights)

    truth <- result$truth
    estimate <- result$estimate
    case_weights <- result$case_weights
  } else if (yardstick_any_missing(truth, estimate, case_weights)) {
    return(NA_real_)
  }

  miss_rate_impl(truth, estimate, event_level)
}
```
:::

Another change from the numeric metric is that a call to `finalize_estimator()` is made. This is the infrastructure that auto-selects the type of estimator to use.

::: {.cell layout-align="center"}

```{.r .cell-code}
data("two_class_example")
miss_rate_vec(two_class_example$truth, two_class_example$predicted)
#> [1] 0.120155
```
:::

What happens if you try and pass in a multiclass result?

::: {.cell layout-align="center"}

```{.r .cell-code}
data("hpc_cv")
fold1 <- filter(hpc_cv, Resample == "Fold01")
miss_rate_vec(fold1$obs, fold1$pred)
#>          F          M          L 
#> 0.06214689 0.00000000 0.00000000
```
:::

This isn't great, as currently multiclass `miss_rate()` isn't supported and it would have been better to throw an error if the `estimator` was not `"binary"`. Currently, `finalize_estimator()` uses its default implementation which selected `"macro"` as the `estimator` since `truth` was a factor with more than 2 classes. When we implement multiclass averaging, this is what you want, but if your metric only works with a binary implementation (or has other specialized multiclass versions), you might want to guard against this.

To fix this, a generic counterpart to `finalize_estimator()`, called `finalize_estimator_internal()`, exists that helps you restrict the input types. If you provide a method to `finalize_estimator_internal()` where the method name is the same as your metric name, and then set the `metric_class` argument in `finalize_estimator()` to be the same thing, you can control how the auto-selection of the `estimator` is handled.

Don't worry about the `metric_dispatcher` argument. This is handled for you and just exists as a dummy argument to dispatch off of.

It is also good practice to call `validate_estimator()` which handles the case where a user passed in the estimator themselves. This validates that the supplied `estimator` is one of the allowed types and error otherwise.

::: {.cell layout-align="center"}

```{.r .cell-code}
finalize_estimator_internal.miss_rate <- function(metric_dispatcher, x, estimator, call) {
  
  validate_estimator(estimator, estimator_override = "binary")
  if (!is.null(estimator)) {
    return(estimator)
  }
  
  lvls <- levels(x)
  if (length(lvls) > 2) {
    stop("A multiclass `truth` input was provided, but only `binary` is supported.")
  } 
  "binary"
}

miss_rate_vec <- function(truth,
                          estimate,
                          estimator = NULL,
                          na_rm = TRUE,
                          case_weights = NULL,
                          event_level = "first",
                          ...) {
  # calls finalize_estimator_internal() internally
  estimator <- finalize_estimator(truth, estimator, metric_class = "miss_rate")
  
  check_class_metric(truth, estimate, case_weights, estimator)
  
  if (na_rm) {
    result <- yardstick_remove_missing(truth, estimate, case_weights)

    truth <- result$truth
    estimate <- result$estimate
    case_weights <- result$case_weights
  } else if (yardstick_any_missing(truth, estimate, case_weights)) {
    return(NA_real_)
  }

  miss_rate_impl(truth, estimate, event_level)
}

# Error thrown by our custom handler
# miss_rate_vec(fold1$obs, fold1$pred)

# Error thrown by validate_estimator()
# miss_rate_vec(fold1$obs, fold1$pred, estimator = "macro")
```
:::

### Supporting multiclass miss rate

Like many other classification metrics such as `precision()` or `recall()`, miss rate does not have a natural multiclass extension, but one can be created using methods such as macro, weighted macro, and micro averaging. If you have not, I encourage you to read `vignette("multiclass", "yardstick")` for more information about how these methods work.

Generally, they require more effort to get right than the binary case, especially if you want to have a performant version. Luckily, a somewhat standard template is used in yardstick and can be used here as well.

Let's first remove the "binary" restriction we created earlier.

::: {.cell layout-align="center"}

```{.r .cell-code}
rm(finalize_estimator_internal.miss_rate)
```
:::

The main changes below are:

-   The binary implementation is moved to `miss_rate_binary()`.

-   `miss_rate_estimator_impl()` is a helper function for switching between binary and multiclass implementations. It also applies the weighting required for multiclass estimators. It is called from `miss_rate_impl()` and also accepts the `estimator` argument using R's function scoping rules.

-   `miss_rate_multiclass()` provides the implementation for the multiclass case. It calculates the true positive and false negative values as vectors with one value per class. For the macro case, it returns a vector of miss rate calculations, and for micro, it first sums the individual pieces and returns a single miss rate calculation. In the macro case, the vector is then weighted appropriately in `miss_rate_estimator_impl()` depending on whether or not it was macro or weighted macro.

::: {.cell layout-align="center"}

```{.r .cell-code}
miss_rate_vec <- function(truth, 
                          estimate, 
                          estimator = NULL, 
                          na_rm = TRUE, 
                          case_weights = NULL,
                          event_level = "first",
                          ...) {
  # calls finalize_estimator_internal() internally
  estimator <- finalize_estimator(truth, estimator, metric_class = "miss_rate")
  
  check_class_metric(truth, estimate, case_weights, estimator)
  
  if (na_rm) {
    result <- yardstick_remove_missing(truth, estimate, case_weights)

    truth <- result$truth
    estimate <- result$estimate
    case_weights <- result$case_weights
  } else if (yardstick_any_missing(truth, estimate, case_weights)) {
    return(NA_real_)
  }

  miss_rate_impl(truth, estimate, estimator, event_level)
}

miss_rate_impl <- function(truth, estimate, estimator, event_level) {
  xtab <- table(estimate, truth)
  # Rather than implement the actual method here, we rely on
  # an *_estimator_impl() function that can handle binary
  # and multiclass cases
  miss_rate_estimator_impl(xtab, estimator, event_level)
}

# This function switches between binary and multiclass implementations
miss_rate_estimator_impl <- function(data, estimator, event_level) {
  if(estimator == "binary") {
    miss_rate_binary(data, event_level)
  } else {
    # Encapsulates the macro, macro weighted, and micro cases
    wt <- get_weights(data, estimator)
    res <- miss_rate_multiclass(data, estimator)
    weighted.mean(res, wt)
  }
}

miss_rate_binary <- function(data, event_level) {
  col <- event_col(data, event_level)
  col2 <- setdiff(colnames(data), col)
  
  tp <- data[col, col]
  fn <- data[col2, col]
  
  fn / (fn + tp)
}

miss_rate_multiclass <- function(data, estimator) {
  
  # We need tp and fn for all classes individually
  # we can get this by taking advantage of the fact
  # that tp + fn = colSums(data)
  tp <- diag(data)
  tpfn <- colSums(data)
  fn <- tpfn - tp
  
  # If using a micro estimator, we sum the individual
  # pieces before performing the miss rate calculation
  if (estimator == "micro") {
    tp <- sum(tp)
    fn <- sum(fn)
  }
  
  # return the vector 
  tp / (tp + fn)
}
```
:::

For the macro case, this separation of weighting from the core implementation might seem strange, but there is good reason for it. Some metrics are combinations of other metrics, and it is nice to be able to reuse code when calculating more complex metrics. For example, `f_meas()` is a combination of `recall()` and `precision()`. When calculating a macro averaged `f_meas()`, the weighting must be applied 1 time, at the very end of the calculation. `recall_multiclass()` and `precision_multiclass()` are defined similarly to how `miss_rate_multiclass()` is defined and returns the unweighted vector of calculations. This means we can directly use this in `f_meas()`, and then weight everything once at the end of that calculation.

Let's try it out now:

::: {.cell layout-align="center"}

```{.r .cell-code}
# two class
miss_rate_vec(two_class_example$truth, two_class_example$predicted)
#> [1] 0.120155

# multiclass
miss_rate_vec(fold1$obs, fold1$pred)
#> [1] 0.5483506
```
:::

#### Data frame implementation

Luckily, the data frame implementation is as simple as the numeric case, we just need to add an extra `estimator` argument and pass that through.

::: {.cell layout-align="center"}

```{.r .cell-code}
miss_rate <- function(data, ...) {
  UseMethod("miss_rate")
}

miss_rate <- new_class_metric(miss_rate, direction = "minimize")

miss_rate.data.frame <- function(data, 
                                 truth, 
                                 estimate, 
                                 estimator = NULL, 
                                 na_rm = TRUE, 
                                 case_weights = NULL,
                                 event_level = "first",
                                 ...) {
  class_metric_summarizer(
    name = "miss_rate",
    fn = miss_rate_vec,
    data = data,
    truth = !!enquo(truth),
    estimate = !!enquo(estimate), 
    estimator = estimator,
    na_rm = na_rm,
    case_weights = !!enquo(case_weights),
    event_level = event_level
  )
}
```
:::

::: {.cell layout-align="center"}

```{.r .cell-code}
# Macro weighted automatically selected
fold1 %>%
  miss_rate(obs, pred)

# Switch to micro
fold1 %>%
  miss_rate(obs, pred, estimator = "micro")

# Macro weighted by resample
hpc_cv %>%
  group_by(Resample) %>%
  miss_rate(obs, pred, estimator = "macro_weighted")

# Error handling
miss_rate(hpc_cv, obs, VF)
```
:::

## Using custom metrics

The `metric_set()` function validates that all metric functions are of the same metric type by checking the class of the function. If any metrics are not of the right class, `metric_set()` fails. By using `new_numeric_metric()` and `new_class_metric()` in the above custom metrics, they work out of the box without any additional adjustments.

::: {.cell layout-align="center"}

```{.r .cell-code}
numeric_mets <- metric_set(mse, rmse)

numeric_mets(solubility_test, solubility, prediction)
#> # A tibble: 2 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 mse     standard       0.521
#> 2 rmse    standard       0.722
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
#>  tibble       3.2.1   2023-03-20 CRAN (R 4.4.0)
#>  tidymodels   1.3.0   2025-02-21 CRAN (R 4.4.1)
#>  tune         1.3.0   2025-02-21 CRAN (R 4.4.1)
#>  workflows    1.2.0   2025-02-19 CRAN (R 4.4.1)
#>  yardstick    1.3.2   2025-01-22 CRAN (R 4.4.1)
#> 
#> ────────────────────────────────────────────────────────────────────
```
:::
