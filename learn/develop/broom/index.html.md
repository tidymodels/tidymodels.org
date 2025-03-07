---
title: "Create your own broom tidier methods"
categories:
  - developer tools
type: learn-subsection
weight: 5
description: | 
  Write tidy(), glance(), and augment() methods for new model objects.
toc: true
toc-depth: 2
include-after-body: ../../../resources.html
---







## Introduction

To use code in this article,  you will need to install the following packages: generics, tidymodels, tidyverse, and usethis.

The broom package provides tools to summarize key information about models in tidy `tibble()`s. The package provides three verbs, or "tidiers," to help make model objects easier to work with:

* `tidy()` summarizes information about model components
* `glance()` reports information about the entire model
* `augment()` adds information about observations to a dataset

Each of the three verbs above are _generic_, in that they do not define a procedure to tidy a given model object, but instead redirect to the relevant _method_ implemented to tidy a specific type of model object. The broom package provides methods for model objects from over 100 modeling packages along with nearly all of the model objects in the stats package that comes with base R. However, for maintainability purposes, the broom package authors now ask that requests for new methods be first directed to the parent package (i.e. the package that supplies the model object) rather than to broom. New methods will generally only be integrated into broom in the case that the requester has already asked the maintainers of the model-owning package to implement tidier methods in the parent package.

We'd like to make implementing external tidier methods as painless as possible. The general process for doing so is:

* re-export the tidier generics
* implement tidying methods
* document the new methods

In this article, we'll walk through each of the above steps in detail, giving examples and pointing out helpful functions when possible.

## Re-export the tidier generics

The first step is to re-export the generic functions for `tidy()`, `glance()`, and/or `augment()`. You could do so from `broom` itself, but we've provided an alternative, much lighter dependency called `generics`.

First you'll need to add the [generics](https://github.com/r-lib/generics) package to `Imports`. We recommend using the [usethis](https://github.com/r-lib/usethis) package for this:


::: {.cell layout-align="center"}

```{.r .cell-code}
usethis::use_package("generics", "Imports")
```
:::


Next, you'll need to re-export the appropriate tidying methods. If you plan to implement a `glance()` method, for example, you can re-export the `glance()` generic by adding the following somewhere inside the `/R` folder of your package:


::: {.cell layout-align="center"}

```{.r .cell-code}
#' @importFrom generics glance
#' @export
generics::glance
```
:::


Oftentimes it doesn't make sense to define one or more of these methods for a particular model. In this case, only implement the methods that do make sense.

::: {.callout-warning}
 Please do not define `tidy()`, `glance()`, or `augment()` generics in your package. This will result in namespace conflicts whenever your package is used along other packages that also export tidying methods. 
:::

## Implement tidying methods

You'll now need to implement specific tidying methods for each of the generics you've re-exported in the above step. For each of `tidy()`, `glance()`, and `augment()`, we'll walk through the big picture, an example, and helpful resources.

In this article, we'll use the base R dataset `trees`, giving the tree girth (in inches), height (in feet), and volume (in cubic feet), to fit an example linear model using the base R `lm()` function. 


::: {.cell layout-align="center"}

```{.r .cell-code}
# load in the trees dataset
data(trees)

# take a look!
str(trees)
#> 'data.frame':	31 obs. of  3 variables:
#>  $ Girth : num  8.3 8.6 8.8 10.5 10.7 10.8 11 11 11.1 11.2 ...
#>  $ Height: num  70 65 63 72 81 83 66 75 80 75 ...
#>  $ Volume: num  10.3 10.3 10.2 16.4 18.8 19.7 15.6 18.2 22.6 19.9 ...

# fit the timber volume as a function of girth and height
trees_model <- lm(Volume ~ Girth + Height, data = trees)
```
:::


Let's take a look at the `summary()` of our `trees_model` fit.


::: {.cell layout-align="center"}

```{.r .cell-code}
summary(trees_model)
#> 
#> Call:
#> lm(formula = Volume ~ Girth + Height, data = trees)
#> 
#> Residuals:
#>     Min      1Q  Median      3Q     Max 
#> -6.4065 -2.6493 -0.2876  2.2003  8.4847 
#> 
#> Coefficients:
#>             Estimate Std. Error t value Pr(>|t|)    
#> (Intercept) -57.9877     8.6382  -6.713 2.75e-07 ***
#> Girth         4.7082     0.2643  17.816  < 2e-16 ***
#> Height        0.3393     0.1302   2.607   0.0145 *  
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Residual standard error: 3.882 on 28 degrees of freedom
#> Multiple R-squared:  0.948,	Adjusted R-squared:  0.9442 
#> F-statistic:   255 on 2 and 28 DF,  p-value: < 2.2e-16
```
:::


This output gives some summary statistics on the residuals (which would be described more fully in an `augment()` output), model coefficients (which, in this case, make up the `tidy()` output), and some model-level summarizations such as RSE, $R^2$, etc. (which make up the `glance()` output.)

### Implementing the `tidy()` method

The `tidy(x, ...)` method will return a tibble where each row contains information about a component of the model. The `x` input is a model object, and the dots (`...`) are an optional argument to supply additional information to any calls inside your method. New `tidy()` methods can take additional arguments, but _must_ include the `x` and `...` arguments to be compatible with the generic function. (For a glossary of currently acceptable additional arguments, see [the end of this article](#glossary).)  Examples of model components include regression coefficients (for regression models), clusters (for classification/clustering models), etc. These `tidy()` methods are useful for inspecting model details and creating custom model visualizations.

Returning to the example of our linear model on timber volume, we'd like to extract information on the model components. In this example, the components are the regression coefficients. After taking a look at the model object and its `summary()`, you might notice that you can extract the regression coefficients as follows:


::: {.cell layout-align="center"}

```{.r .cell-code}
summary(trees_model)$coefficients
#>                Estimate Std. Error   t value     Pr(>|t|)
#> (Intercept) -57.9876589  8.6382259 -6.712913 2.749507e-07
#> Girth         4.7081605  0.2642646 17.816084 8.223304e-17
#> Height        0.3392512  0.1301512  2.606594 1.449097e-02
```
:::


This object contains the model coefficients as a table, where the information giving which coefficient is being described in each row is given in the row names. Converting to a tibble where the row names are contained in a column, you might write:


::: {.cell layout-align="center"}

```{.r .cell-code}
trees_model_tidy <- summary(trees_model)$coefficients %>% 
  as_tibble(rownames = "term")

trees_model_tidy
#> # A tibble: 3 × 5
#>   term        Estimate `Std. Error` `t value` `Pr(>|t|)`
#>   <chr>          <dbl>        <dbl>     <dbl>      <dbl>
#> 1 (Intercept)  -58.0          8.64      -6.71   2.75e- 7
#> 2 Girth          4.71         0.264     17.8    8.22e-17
#> 3 Height         0.339        0.130      2.61   1.45e- 2
```
:::


The broom package standardizes common column names used to describe coefficients. In this case, the column names are:


::: {.cell layout-align="center"}

```{.r .cell-code}
colnames(trees_model_tidy) <- c("term", "estimate", "std.error", "statistic", "p.value")
```
:::


A glossary giving the currently acceptable column names outputted by `tidy()` methods can be found [at the end of this article](#glossary). As a rule of thumb, column names resulting from `tidy()` methods should be all lowercase and contain only alphanumerics or periods (though there are plenty of exceptions).

Finally, it is common for `tidy()` methods to include an option to calculate confidence/credible intervals for each component based on the model, when possible. In this example, the `confint()` function can be used to calculate confidence intervals from a model object resulting from `lm()`:


::: {.cell layout-align="center"}

```{.r .cell-code}
confint(trees_model)
#>                    2.5 %      97.5 %
#> (Intercept) -75.68226247 -40.2930554
#> Girth         4.16683899   5.2494820
#> Height        0.07264863   0.6058538
```
:::


With these considerations in mind, a reasonable `tidy()` method for `lm()` might look something like:


::: {.cell layout-align="center"}

```{.r .cell-code}
tidy.lm <- function(x, conf.int = FALSE, conf.level = 0.95, ...) {
  
  result <- summary(x)$coefficients %>%
    tibble::as_tibble(rownames = "term") %>%
    dplyr::rename(estimate = Estimate,
                  std.error = `Std. Error`,
                  statistic = `t value`,
                  p.value = `Pr(>|t|)`)
  
  if (conf.int) {
    ci <- confint(x, level = conf.level)
    result <- dplyr::left_join(result, ci, by = "term")
  }
  
  result
}
```
:::


::: {.callout-note}
 If you're interested, the actual `tidy.lm()` source can be found [here](https://github.com/tidymodels/broom/blob/master/R/stats-lm-tidiers.R)! It's not too different from the version above except for some argument checking and additional columns. 
:::

With this method exported, then, if a user calls `tidy(fit)`, where `fit` is an output from `lm()`, the `tidy()` generic would "redirect" the call to the `tidy.lm()` function above.

Some things to keep in mind while writing your `tidy()` method:

* Sometimes a model will have several different types of components. For example, in mixed models, there is different information associated with fixed effects and random effects. Since this information doesn't have the same interpretation, it doesn't make sense to summarize the fixed and random effects in the same table. In cases like this you should add an argument that allows the user to specify which type of information they want. For example, you might implement an interface along the lines of:


::: {.cell layout-align="center"}

```{.r .cell-code}
model <- mixed_model(...)
tidy(model, effects = "fixed")
tidy(model, effects = "random")
```
:::


* How are missing values encoded in the model object and its `summary()`? Ensure that rows are included even when the associated model component is missing or rank deficient.
* Are there other measures specific to each component that could reasonably be expected to be included in their summarizations? Some common arguments to `tidy()` methods include:
  - `conf.int`: A logical indicating whether or not to calculate confidence/credible intervals. This should default to `FALSE`.
  - `conf.level`: The confidence level to use for the interval when `conf.int = TRUE`. Typically defaults to `.95`.
  - `exponentiate`: A logical indicating whether or not model terms should be presented on an exponential scale (typical for logistic regression).

### Implementing the `glance()` method

`glance()` returns a one-row tibble providing model-level summarizations (e.g. goodness of fit measures and related statistics). This is useful to check for model misspecification and to compare many models. Again, the `x` input is a model object, and the `...` is an optional argument to supply additional information to any calls inside your method. New `glance()` methods can also take additional arguments and _must_ include the `x` and `...` arguments. (For a glossary of currently acceptable additional arguments, see [the end of this article](#glossary).)

Returning to the `trees_model` example, we could pull out the $R^2$ value with the following code:


::: {.cell layout-align="center"}

```{.r .cell-code}
summary(trees_model)$r.squared
#> [1] 0.94795
```
:::


Similarly, for the adjusted $R^2$:


::: {.cell layout-align="center"}

```{.r .cell-code}
summary(trees_model)$adj.r.squared
#> [1] 0.9442322
```
:::


Unfortunately, for many model objects, the extraction of model-level information is largely a manual process. You will likely need to build a `tibble()` element-by-element by subsetting the `summary()` object repeatedly. The `with()` function, however, can help make this process a bit less tedious by evaluating expressions inside of the `summary(trees_model)` environment. To grab those those same two model elements from above using `with()`:


::: {.cell layout-align="center"}

```{.r .cell-code}
with(summary(trees_model),
     tibble::tibble(r.squared = r.squared,
                    adj.r.squared = adj.r.squared))
#> # A tibble: 1 × 2
#>   r.squared adj.r.squared
#>       <dbl>         <dbl>
#> 1     0.948         0.944
```
:::


A reasonable `glance()` method for `lm()`, then, might look something like:


::: {.cell layout-align="center"}

```{.r .cell-code}
glance.lm <- function(x, ...) {
  with(
    summary(x),
    tibble::tibble(
      r.squared = r.squared,
      adj.r.squared = adj.r.squared,
      sigma = sigma,
      statistic = fstatistic["value"],
      p.value = pf(
        fstatistic["value"],
        fstatistic["numdf"],
        fstatistic["dendf"],
        lower.tail = FALSE
      ),
      df = fstatistic["numdf"],
      logLik = as.numeric(stats::logLik(x)),
      AIC = stats::AIC(x),
      BIC = stats::BIC(x),
      deviance = stats::deviance(x),
      df.residual = df.residual(x),
      nobs = stats::nobs(x)
    )
  )
}
```
:::


::: {.callout-note}
This is the actual definition of `glance.lm()` provided by broom! 
:::

Some things to keep in mind while writing `glance()` methods:
* Output should not include the name of the modeling function or any arguments given to the modeling function.
* In some cases, you may wish to provide model-level diagnostics not returned by the original object. For example, the above `glance.lm()` calculates `AIC` and `BIC` from the model fit. If these are easy to compute, feel free to add them. However, tidier methods are generally not an appropriate place to implement complex or time consuming calculations.
* The `glance` method should always return the same columns in the same order when given an object of a given model class. If a summary metric (such as `AIC`) is not defined in certain circumstances, use `NA`.

### Implementing the `augment()` method

`augment()` methods add columns to a dataset containing information such as fitted values, residuals or cluster assignments. All columns added to a dataset have a `.` prefix to prevent existing columns from being overwritten. (Currently acceptable column names are given in [the glossary](#glossary).) The `x` and `...` arguments share their meaning with the two functions described above. `augment` methods also optionally accept a `data` argument that is a `data.frame` (or `tibble`) to add observation-level information to, returning a `tibble` object with the same number of rows as `data`. Many `augment()` methods also accept a `newdata` argument, following the same conventions as the `data` argument, except with the underlying assumption that the model has not "seen" the data yet. As a result, `newdata` arguments need not contain the response columns in `data`. Only one of `data` or `newdata` should be supplied. A full glossary of acceptable arguments to `augment()` methods can be found at [the end of this article](#glossary).

If a `data` argument is not specified, `augment()` should try to reconstruct the original data as much as possible from the model object. This may not always be possible, and often it will not be possible to recover columns not used by the model.

With this is mind, we can look back to our `trees_model` example. For one, the `model` element inside of the `trees_model` object will allow us to recover the original data:


::: {.cell layout-align="center"}

```{.r .cell-code}
trees_model$model
#>    Volume Girth Height
#> 1    10.3   8.3     70
#> 2    10.3   8.6     65
#> 3    10.2   8.8     63
#> 4    16.4  10.5     72
#> 5    18.8  10.7     81
#> 6    19.7  10.8     83
#> 7    15.6  11.0     66
#> 8    18.2  11.0     75
#> 9    22.6  11.1     80
#> 10   19.9  11.2     75
#> 11   24.2  11.3     79
#> 12   21.0  11.4     76
#> 13   21.4  11.4     76
#> 14   21.3  11.7     69
#> 15   19.1  12.0     75
#> 16   22.2  12.9     74
#> 17   33.8  12.9     85
#> 18   27.4  13.3     86
#> 19   25.7  13.7     71
#> 20   24.9  13.8     64
#> 21   34.5  14.0     78
#> 22   31.7  14.2     80
#> 23   36.3  14.5     74
#> 24   38.3  16.0     72
#> 25   42.6  16.3     77
#> 26   55.4  17.3     81
#> 27   55.7  17.5     82
#> 28   58.3  17.9     80
#> 29   51.5  18.0     80
#> 30   51.0  18.0     80
#> 31   77.0  20.6     87
```
:::


Similarly, the fitted values and residuals can be accessed with the following code:


::: {.cell layout-align="center"}

```{.r .cell-code}
head(trees_model$fitted.values)
#>         1         2         3         4         5         6 
#>  4.837660  4.553852  4.816981 15.874115 19.869008 21.018327
head(trees_model$residuals)
#>          1          2          3          4          5          6 
#>  5.4623403  5.7461484  5.3830187  0.5258848 -1.0690084 -1.3183270
```
:::


As with `glance()` methods, it's fine (and encouraged!) to include common metrics associated with observations if they are not computationally intensive to compute. A common metric associated with linear models, for example, is the standard error of fitted values:


::: {.cell layout-align="center"}

```{.r .cell-code}
se.fit <- predict(trees_model, newdata = trees, se.fit = TRUE)$se.fit %>%
  unname()

head(se.fit)
#> [1] 1.3211285 1.4893775 1.6325024 0.9444212 1.3484251 1.5319772
```
:::


Thus, a reasonable `augment()` method for `lm` might look something like this:


::: {.cell layout-align="center"}

```{.r .cell-code}
augment.lm <- function(x, data = x$model, newdata = NULL, ...) {
  if (is.null(newdata)) {
    dplyr::bind_cols(tibble::as_tibble(data),
                     tibble::tibble(.fitted = x$fitted.values,
                                    .se.fit = predict(x, 
                                                      newdata = data, 
                                                      se.fit = TRUE)$se.fit,
                                   .resid =  x$residuals))
  } else {
    predictions <- predict(x, newdata = newdata, se.fit = TRUE)
    dplyr::bind_cols(tibble::as_tibble(newdata),
                     tibble::tibble(.fitted = predictions$fit,
                                    .se.fit = predictions$se.fit))
  }
}
```
:::


Some other things to keep in mind while writing `augment()` methods:
* The `newdata` argument should default to `NULL`. Users should only ever specify one of `data` or `newdata`. Providing both `data` and `newdata` should result in an error. The `newdata` argument should accept both `data.frame`s and `tibble`s.
* Data given to the `data` argument must have both the original predictors and the original response. Data given to the `newdata` argument only needs to have the original predictors. This is important because there may be important information associated with training data that is not associated with test data. This means that the `original_data` object in `augment(model, data = original_data)` should provide `.fitted` and `.resid` columns (in most cases), whereas `test_data` in `augment(model, data = test_data)` only needs a `.fitted` column, even if the response is present in `test_data`.
* If the `data` or `newdata` is specified as a `data.frame` with rownames, `augment` should return them in a column called `.rownames`.
* For observations where no fitted values or summaries are available (where there's missing data, for example), return `NA`.
* *The `augment()` method should always return as many rows as were in `data` or `newdata`*, depending on which is supplied

::: {.callout-note}
The recommended interface and functionality for `augment()` methods may change soon. 
:::

## Document the new methods

The only remaining step is to integrate the new methods into the parent package! To do so, just drop the methods into a `.R` file inside of the `/R` folder and document them using roxygen2. If you're unfamiliar with the process of documenting objects, you can read more about it [here](http://r-pkgs.had.co.nz/man.html). Here's an example of how our `tidy.lm()` method might be documented:


::: {.cell layout-align="center"}

```{.r .cell-code}
#' Tidy a(n) lm object
#'
#' @param x A `lm` object.
#' @param conf.int Logical indicating whether or not to include 
#'   a confidence interval in the tidied output. Defaults to FALSE.
#' @param conf.level The confidence level to use for the confidence 
#'   interval if conf.int = TRUE. Must be strictly greater than 0 
#'   and less than 1. Defaults to 0.95, which corresponds to a 
#'   95 percent confidence interval.
#' @param ... Unused, included for generic consistency only.
#' @return A tidy [tibble::tibble()] summarizing component-level
#'   information about the model
#'
#' @examples
#' # load the trees dataset
#' data(trees)
#' 
#' # fit a linear model on timber volume
#' trees_model <- lm(Volume ~ Girth + Height, data = trees)
#'
#' # summarize model coefficients in a tidy tibble!
#' tidy(trees_model)
#'
#' @export
tidy.lm <- function(x, conf.int = FALSE, conf.level = 0.95, ...) {

  # ... the rest of the function definition goes here!
```
:::


Once you've documented each of your new methods and executed `devtools::document()`, you're done! Congrats on implementing your own broom tidier methods for a new model object!

## Glossaries






### Arguments

Tidier methods have a standardized set of acceptable argument and output column names. The currently acceptable argument names by tidier method are:


::: {.cell layout-align="center"}
::: {.cell-output-display}


```{=html}
<div class="datatables html-widget html-fill-item" id="htmlwidget-1637191b85587329b8db" style="width:100%;height:auto;"></div>
<script type="application/json" data-for="htmlwidget-1637191b85587329b8db">{"x":{"filter":"top","vertical":false,"filterHTML":"<tr>\n  <td data-type=\"factor\" style=\"vertical-align: top;\">\n    <div class=\"form-group has-feedback\" style=\"margin-bottom: auto;\">\n      <input type=\"search\" placeholder=\"All\" class=\"form-control\" style=\"width: 100%;\"/>\n      <span class=\"glyphicon glyphicon-remove-circle form-control-feedback\"><\/span>\n    <\/div>\n    <div style=\"width: 100%; display: none;\">\n      <select multiple=\"multiple\" style=\"width: 100%;\" data-options=\"[&quot;augment&quot;,&quot;glance&quot;,&quot;tidy&quot;]\"><\/select>\n    <\/div>\n  <\/td>\n  <td data-type=\"character\" style=\"vertical-align: top;\">\n    <div class=\"form-group has-feedback\" style=\"margin-bottom: auto;\">\n      <input type=\"search\" placeholder=\"All\" class=\"form-control\" style=\"width: 100%;\"/>\n      <span class=\"glyphicon glyphicon-remove-circle form-control-feedback\"><\/span>\n    <\/div>\n  <\/td>\n<\/tr>","data":[["tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","glance","glance","glance","glance","glance","glance","augment","augment","augment","augment","augment","augment","augment","augment"],["alpha","boot_se","by_class","col.names","component","conf.int","conf.level","conf.method","conf.type","diagonal","droppars","effects","ess","estimate.method","exponentiate","fe","include_studies","instruments","intervals","matrix","measure","na.rm","object","p.values","par_type","parameters","parametric","pars","prob","region","return_zeros","rhat","robust","scales","se.type","strata","test","trim","upper","deviance","diagnostics","looic","mcmc","test","x","conf.level","data","interval","newdata","se_fit","type.predict","type.residuals","weights"]],"container":"<table class=\"cell-border stripe\">\n  <thead>\n    <tr>\n      <th>Method<\/th>\n      <th>Argument<\/th>\n    <\/tr>\n  <\/thead>\n<\/table>","options":{"pageLength":5,"columnDefs":[{"name":"Method","targets":0},{"name":"Argument","targets":1}],"order":[],"autoWidth":false,"orderClasses":false,"orderCellsTop":true,"lengthMenu":[5,10,25,50,100]}},"evals":[],"jsHooks":[]}</script>
```


:::
:::


### Column Names

The currently acceptable column names by tidier method are:


::: {.cell layout-align="center"}
::: {.cell-output-display}


```{=html}
<div class="datatables html-widget html-fill-item" id="htmlwidget-b743132a6c702acea3c2" style="width:100%;height:auto;"></div>
<script type="application/json" data-for="htmlwidget-b743132a6c702acea3c2">{"x":{"filter":"top","vertical":false,"filterHTML":"<tr>\n  <td data-type=\"factor\" style=\"vertical-align: top;\">\n    <div class=\"form-group has-feedback\" style=\"margin-bottom: auto;\">\n      <input type=\"search\" placeholder=\"All\" class=\"form-control\" style=\"width: 100%;\"/>\n      <span class=\"glyphicon glyphicon-remove-circle form-control-feedback\"><\/span>\n    <\/div>\n    <div style=\"width: 100%; display: none;\">\n      <select multiple=\"multiple\" style=\"width: 100%;\" data-options=\"[&quot;augment&quot;,&quot;glance&quot;,&quot;tidy&quot;]\"><\/select>\n    <\/div>\n  <\/td>\n  <td data-type=\"character\" style=\"vertical-align: top;\">\n    <div class=\"form-group has-feedback\" style=\"margin-bottom: auto;\">\n      <input type=\"search\" placeholder=\"All\" class=\"form-control\" style=\"width: 100%;\"/>\n      <span class=\"glyphicon glyphicon-remove-circle form-control-feedback\"><\/span>\n    <\/div>\n  <\/td>\n<\/tr>","data":[["tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","tidy","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","glance","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment","augment"],["acf","adj.p.value","alternative","at.value","at.variable","atmean","autocorrelation","bias","ci.width","class","cluster","coef.type","column1","column2","comp","comparison","component","conf.high","conf.low","contrast","cumulative","cutoff","delta","den.df","denominator","dev.ratio","df","distance","estimate","estimate1","estimate2","event","exp","expected","fpr","freq","GCV","group","group1","group2","index","item1","item2","kendall_score","lag","lambda","letters","lhs","logLik","mcmc.error","mean","meansq","method","n","N","n.censor","n.event","n.risk","null.value","num.df","nzero","obs","op","outcome","p","p.value","p.value.Sargan","p.value.weakinst","p.value.Wu.Hausman","parameter","PC","percent","power","proportion","pyears","response","rhs","robust.se","row","scale","sd","series","sig.level","size","spec","state","statistic","statistic.Sargan","statistic.weakinst","statistic.Wu.Hausman","std_estimate","std.all","std.dev","std.error","std.lv","std.nox","step","strata","stratum","study","sumsq","tau","term","time","tpr","type","uniqueness","value","var_kendall_score","variable","variance","withinss","y.level","y.value","z","adj.r.squared","agfi","AIC","AICc","alpha","alternative","autocorrelation","avg.silhouette.width","betweenss","BIC","cfi","chi.squared","chisq","cochran.qe","cochran.qm","conf.high","conf.low","converged","convergence","crit","cv.crit","den.df","deviance","df","df.null","df.residual","dw.original","dw.transformed","edf","estimator","events","finTol","function.count","G","g.squared","gamma","gradient.count","H","h.squared","hypvol","i.squared","independence","isConv","iter","iterations","kHKB","kLW","lag.order","lambda","lambda.1se","lambda.min","lambdaGCV","logLik","max.cluster.size","max.hazard","max.time","maxit","MCMC.burnin","MCMC.interval","MCMC.samplesize","measure","median","method","min.hazard","min.time","missing_method","model","n","n.clusters","n.factors","n.max","n.start","nevent","nexcluded","ngroups","nobs","norig","npar","npasses","null.deviance","nulldev","num.df","number.interaction","offtable","p.value","p.value.cochran.qe","p.value.cochran.qm","p.value.original","p.value.Sargan","p.value.transformed","p.value.weak.instr","p.value.Wu.Hausman","parameter","pen.crit","power","power.reached","pseudo.r.squared","r.squared","records","residual.deviance","rho","rho2","rho20","rmean","rmean.std.error","rmsea","rmsea.conf.high","rscore","score","sigma","sigma2_j","spar","srmr","statistic","statistic.Sargan","statistic.weak.instr","statistic.Wu.Hausman","tau","tau.squared","tau.squared.se","theta","timepoints","tli","tot.withinss","total","total.variance","totss","value","within.r.squared",".class",".cluster",".cochran.qe.loo",".col.prop",".conf.high",".conf.low",".cooksd",".cov.ratio",".cred.high",".cred.low",".dffits",".expected",".fitted",".fitted_j_0",".fitted_j_1",".hat",".lower",".moderator",".moderator.level",".observed",".probability",".prop",".remainder",".resid",".resid_j_0",".resid_j_1",".row.prop",".rownames",".se.fit",".seasadj",".seasonal",".sigma",".std.resid",".tau",".tau.squared.loo",".trend",".uncertainty",".upper",".weight"]],"container":"<table class=\"cell-border stripe\">\n  <thead>\n    <tr>\n      <th>Method<\/th>\n      <th>Column<\/th>\n    <\/tr>\n  <\/thead>\n<\/table>","options":{"pageLength":5,"columnDefs":[{"name":"Method","targets":0},{"name":"Column","targets":1}],"order":[],"autoWidth":false,"orderClasses":false,"orderCellsTop":true,"lengthMenu":[5,10,25,50,100]}},"evals":[],"jsHooks":[]}</script>
```


:::
:::


The [alexpghayes/modeltests](https://github.com/alexpghayes/modeltests) package provides unit testing infrastructure to check your new tidier methods. Please file an issue there to request new arguments/columns to be added to the glossaries!

## Session information {#session-info}


::: {.cell layout-align="center"}

```
#> ─ Session info ─────────────────────────────────────────────────────
#>  setting  value
#>  version  R version 4.4.0 (2024-04-24)
#>  os       macOS Sonoma 14.4.1
#>  system   aarch64, darwin20
#>  ui       X11
#>  language (EN)
#>  collate  en_US.UTF-8
#>  ctype    en_US.UTF-8
#>  tz       America/Los_Angeles
#>  date     2024-06-26
#>  pandoc   3.1.1 @ /Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/ (via rmarkdown)
#> 
#> ─ Packages ─────────────────────────────────────────────────────────
#>  package    * version date (UTC) lib source
#>  broom      * 1.0.6   2024-05-17 [1] CRAN (R 4.4.0)
#>  dials      * 1.2.1   2024-02-22 [1] CRAN (R 4.4.0)
#>  dplyr      * 1.1.4   2023-11-17 [1] CRAN (R 4.4.0)
#>  generics   * 0.1.3   2022-07-05 [1] CRAN (R 4.4.0)
#>  ggplot2    * 3.5.1   2024-04-23 [1] CRAN (R 4.4.0)
#>  infer      * 1.0.7   2024-03-25 [1] CRAN (R 4.4.0)
#>  parsnip    * 1.2.1   2024-03-22 [1] CRAN (R 4.4.0)
#>  purrr      * 1.0.2   2023-08-10 [1] CRAN (R 4.4.0)
#>  recipes    * 1.0.10  2024-02-18 [1] CRAN (R 4.4.0)
#>  rlang        1.1.4   2024-06-04 [1] CRAN (R 4.4.0)
#>  rsample    * 1.2.1   2024-03-25 [1] CRAN (R 4.4.0)
#>  tibble     * 3.2.1   2023-03-20 [1] CRAN (R 4.4.0)
#>  tidymodels * 1.2.0   2024-03-25 [1] CRAN (R 4.4.0)
#>  tidyverse  * 2.0.0   2023-02-22 [1] CRAN (R 4.4.0)
#>  tune       * 1.2.1   2024-04-18 [1] CRAN (R 4.4.0)
#>  workflows  * 1.1.4   2024-02-19 [1] CRAN (R 4.4.0)
#>  yardstick  * 1.3.1   2024-03-21 [1] CRAN (R 4.4.0)
#> 
#>  [1] /Library/Frameworks/R.framework/Versions/4.4-arm64/Resources/library
#> 
#> ────────────────────────────────────────────────────────────────────
```
:::
