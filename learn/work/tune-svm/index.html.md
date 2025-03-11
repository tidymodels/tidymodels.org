---
title: "Model tuning via grid search"
categories:
  - tuning
  - SVMs
  - classification
type: learn-subsection
weight: 1
description: | 
  Choose hyperparameters for a model by training on a grid of many possible parameter values.
toc: true
toc-depth: 2
include-after-body: ../../../resources.html
---









## Introduction

To use code in this article,  you will need to install the following packages: kernlab, mlbench, and tidymodels.

This article demonstrates how to tune a model using grid search. Many models have **hyperparameters** that can't be learned directly from a single data set when training the model. Instead, we can train many models in a grid of possible hyperparameter values and see which ones turn out best. 

## Example data

To demonstrate model tuning, we'll use the Ionosphere data in the mlbench package:




::: {.cell layout-align="center"}

```{.r .cell-code}
library(tidymodels)
library(mlbench)
data(Ionosphere)
```
:::




From `?Ionosphere`:

> This radar data was collected by a system in Goose Bay, Labrador. This system consists of a phased array of 16 high-frequency antennas with a total transmitted power on the order of 6.4 kilowatts. See the paper for more details. The targets were free electrons in the ionosphere. "good" radar returns are those showing evidence of some type of structure in the ionosphere. "bad" returns are those that do not; their signals pass through the ionosphere.

> Received signals were processed using an autocorrelation function whose arguments are the time of a pulse and the pulse number. There were 17 pulse numbers for the Goose Bay system. Instances in this databse are described by 2 attributes per pulse number, corresponding to the complex values returned by the function resulting from the complex electromagnetic signal. See cited below for more details.

There are 43 predictors and a factor outcome. Two of the predictors are factors (`V1` and `V2`) and the rest are numeric variables that have been scaled to a range of -1 to 1. Note that the two factor predictors have sparse distributions:




::: {.cell layout-align="center"}

```{.r .cell-code}
table(Ionosphere$V1)
#> 
#>   0   1 
#>  38 313
table(Ionosphere$V2)
#> 
#>   0 
#> 351
```
:::




There's no point of putting `V2` into any model since is is a zero-variance predictor. `V1` is not but it _could_ be if the resampling process ends up sampling all of the same value. Is this an issue? It might be since the standard R formula infrastructure fails when there is only a single observed value:




::: {.cell layout-align="center"}

```{.r .cell-code}
glm(Class ~ ., data = Ionosphere, family = binomial)

# Surprisingly, this doesn't help: 

glm(Class ~ . - V2, data = Ionosphere, family = binomial)
```
:::




Let's remove these two problematic variables:




::: {.cell layout-align="center"}

```{.r .cell-code}
Ionosphere <- Ionosphere %>% select(-V1, -V2)
```
:::




## Inputs for the search

To demonstrate, we'll fit a radial basis function support vector machine to these data and tune the SVM cost parameter and the $\sigma$ parameter in the kernel function:




::: {.cell layout-align="center"}

```{.r .cell-code}
svm_mod <-
  svm_rbf(cost = tune(), rbf_sigma = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")
```
:::




In this article, tuning will be demonstrated in two ways, using:

- a standard R formula, and 
- a recipe.

Let's create a simple recipe here:




::: {.cell layout-align="center"}

```{.r .cell-code}
iono_rec <-
  recipe(Class ~ ., data = Ionosphere)  %>%
  # remove any zero variance predictors
  step_zv(all_predictors()) %>% 
  # remove any linear combinations
  step_lincomb(all_numeric())
```
:::




The only other required item for tuning is a resampling strategy as defined by an rsample object. Let's demonstrate using basic bootstrapping:




::: {.cell layout-align="center"}

```{.r .cell-code}
set.seed(4943)
iono_rs <- bootstraps(Ionosphere, times = 30)
```
:::




## Optional inputs

An _optional_ step for model tuning is to specify which metrics should be computed using the out-of-sample predictions. For classification, the default is to calculate the log-likelihood statistic and overall accuracy. Instead of the defaults, the area under the ROC curve will be used. To do this, a yardstick package function can be used to create a metric set:




::: {.cell layout-align="center"}

```{.r .cell-code}
roc_vals <- metric_set(roc_auc)
```
:::




If no grid or parameters are provided, a set of 10 hyperparameters are created using a space-filling design (via a Latin hypercube). A grid can be given in a data frame where the parameters are in columns and parameter combinations are in rows. Here, the default will be used.

Also, a control object can be passed that specifies different aspects of the search. Here, the verbose option is turned off and the option to save the out-of-sample predictions is turned on. 




::: {.cell layout-align="center"}

```{.r .cell-code}
ctrl <- control_grid(verbose = FALSE, save_pred = TRUE)
```
:::




## Executing with a formula

First, we can use the formula interface:




::: {.cell layout-align="center"}

```{.r .cell-code}
set.seed(35)
formula_res <-
  svm_mod %>% 
  tune_grid(
    Class ~ .,
    resamples = iono_rs,
    metrics = roc_vals,
    control = ctrl
  )
formula_res
#> # Tuning results
#> # Bootstrap sampling 
#> # A tibble: 30 × 5
#>    splits            id          .metrics          .notes           .predictions
#>    <list>            <chr>       <list>            <list>           <list>      
#>  1 <split [351/120]> Bootstrap01 <tibble [10 × 6]> <tibble [0 × 3]> <tibble>    
#>  2 <split [351/130]> Bootstrap02 <tibble [10 × 6]> <tibble [0 × 3]> <tibble>    
#>  3 <split [351/137]> Bootstrap03 <tibble [10 × 6]> <tibble [0 × 3]> <tibble>    
#>  4 <split [351/141]> Bootstrap04 <tibble [10 × 6]> <tibble [0 × 3]> <tibble>    
#>  5 <split [351/131]> Bootstrap05 <tibble [10 × 6]> <tibble [0 × 3]> <tibble>    
#>  6 <split [351/131]> Bootstrap06 <tibble [10 × 6]> <tibble [0 × 3]> <tibble>    
#>  7 <split [351/127]> Bootstrap07 <tibble [10 × 6]> <tibble [0 × 3]> <tibble>    
#>  8 <split [351/123]> Bootstrap08 <tibble [10 × 6]> <tibble [0 × 3]> <tibble>    
#>  9 <split [351/131]> Bootstrap09 <tibble [10 × 6]> <tibble [0 × 3]> <tibble>    
#> 10 <split [351/117]> Bootstrap10 <tibble [10 × 6]> <tibble [0 × 3]> <tibble>    
#> # ℹ 20 more rows
```
:::




The `.metrics` column contains tibbles of the performance metrics for each tuning parameter combination:




::: {.cell layout-align="center"}

```{.r .cell-code}
formula_res %>% 
  select(.metrics) %>% 
  slice(1) %>% 
  pull(1)
#> [[1]]
#> # A tibble: 10 × 6
#>         cost     rbf_sigma .metric .estimator .estimate .config              
#>        <dbl>         <dbl> <chr>   <chr>          <dbl> <chr>                
#>  1  0.000977 0.000000215   roc_auc binary         0.838 Preprocessor1_Model01
#>  2  0.00310  0.00599       roc_auc binary         0.942 Preprocessor1_Model02
#>  3  0.00984  0.0000000001  roc_auc binary         0.815 Preprocessor1_Model03
#>  4  0.0312   0.00000278    roc_auc binary         0.832 Preprocessor1_Model04
#>  5  0.0992   0.0774        roc_auc binary         0.968 Preprocessor1_Model05
#>  6  0.315    0.00000000129 roc_auc binary         0.830 Preprocessor1_Model06
#>  7  1        0.0000359     roc_auc binary         0.837 Preprocessor1_Model07
#>  8  3.17     1             roc_auc binary         0.974 Preprocessor1_Model08
#>  9 10.1      0.0000000167  roc_auc binary         0.832 Preprocessor1_Model09
#> 10 32        0.000464      roc_auc binary         0.861 Preprocessor1_Model10
```
:::




To get the final resampling estimates, the `collect_metrics()` function can be used on the grid object:




::: {.cell layout-align="center"}

```{.r .cell-code}
estimates <- collect_metrics(formula_res)
estimates
#> # A tibble: 10 × 8
#>         cost     rbf_sigma .metric .estimator  mean     n std_err .config       
#>        <dbl>         <dbl> <chr>   <chr>      <dbl> <int>   <dbl> <chr>         
#>  1  0.000977 0.000000215   roc_auc binary     0.871    30 0.00516 Preprocessor1…
#>  2  0.00310  0.00599       roc_auc binary     0.959    30 0.00290 Preprocessor1…
#>  3  0.00984  0.0000000001  roc_auc binary     0.822    30 0.00718 Preprocessor1…
#>  4  0.0312   0.00000278    roc_auc binary     0.871    30 0.00531 Preprocessor1…
#>  5  0.0992   0.0774        roc_auc binary     0.970    30 0.00261 Preprocessor1…
#>  6  0.315    0.00000000129 roc_auc binary     0.857    30 0.00624 Preprocessor1…
#>  7  1        0.0000359     roc_auc binary     0.873    30 0.00533 Preprocessor1…
#>  8  3.17     1             roc_auc binary     0.971    30 0.00248 Preprocessor1…
#>  9 10.1      0.0000000167  roc_auc binary     0.871    30 0.00534 Preprocessor1…
#> 10 32        0.000464      roc_auc binary     0.927    30 0.00484 Preprocessor1…
```
:::




The top combinations are:




::: {.cell layout-align="center"}

```{.r .cell-code}
show_best(formula_res, metric = "roc_auc")
#> # A tibble: 5 × 8
#>       cost rbf_sigma .metric .estimator  mean     n std_err .config             
#>      <dbl>     <dbl> <chr>   <chr>      <dbl> <int>   <dbl> <chr>               
#> 1  3.17    1         roc_auc binary     0.971    30 0.00248 Preprocessor1_Model…
#> 2  0.0992  0.0774    roc_auc binary     0.970    30 0.00261 Preprocessor1_Model…
#> 3  0.00310 0.00599   roc_auc binary     0.959    30 0.00290 Preprocessor1_Model…
#> 4 32       0.000464  roc_auc binary     0.927    30 0.00484 Preprocessor1_Model…
#> 5  1       0.0000359 roc_auc binary     0.873    30 0.00533 Preprocessor1_Model…
```
:::




##  Executing with a recipe

Next, we can use the same syntax but pass a *recipe* in as the pre-processor argument:




::: {.cell layout-align="center"}

```{.r .cell-code}
set.seed(325)
recipe_res <-
  svm_mod %>% 
  tune_grid(
    iono_rec,
    resamples = iono_rs,
    metrics = roc_vals,
    control = ctrl
  )
recipe_res
#> # Tuning results
#> # Bootstrap sampling 
#> # A tibble: 30 × 5
#>    splits            id          .metrics          .notes           .predictions
#>    <list>            <chr>       <list>            <list>           <list>      
#>  1 <split [351/120]> Bootstrap01 <tibble [10 × 6]> <tibble [0 × 3]> <tibble>    
#>  2 <split [351/130]> Bootstrap02 <tibble [10 × 6]> <tibble [0 × 3]> <tibble>    
#>  3 <split [351/137]> Bootstrap03 <tibble [10 × 6]> <tibble [0 × 3]> <tibble>    
#>  4 <split [351/141]> Bootstrap04 <tibble [10 × 6]> <tibble [0 × 3]> <tibble>    
#>  5 <split [351/131]> Bootstrap05 <tibble [10 × 6]> <tibble [0 × 3]> <tibble>    
#>  6 <split [351/131]> Bootstrap06 <tibble [10 × 6]> <tibble [0 × 3]> <tibble>    
#>  7 <split [351/127]> Bootstrap07 <tibble [10 × 6]> <tibble [0 × 3]> <tibble>    
#>  8 <split [351/123]> Bootstrap08 <tibble [10 × 6]> <tibble [0 × 3]> <tibble>    
#>  9 <split [351/131]> Bootstrap09 <tibble [10 × 6]> <tibble [0 × 3]> <tibble>    
#> 10 <split [351/117]> Bootstrap10 <tibble [10 × 6]> <tibble [0 × 3]> <tibble>    
#> # ℹ 20 more rows
```
:::




The best setting here is:




::: {.cell layout-align="center"}

```{.r .cell-code}
show_best(recipe_res, metric = "roc_auc")
#> # A tibble: 5 × 8
#>       cost rbf_sigma .metric .estimator  mean     n std_err .config             
#>      <dbl>     <dbl> <chr>   <chr>      <dbl> <int>   <dbl> <chr>               
#> 1  3.17    1         roc_auc binary     0.971    30 0.00248 Preprocessor1_Model…
#> 2  0.0992  0.0774    roc_auc binary     0.970    30 0.00261 Preprocessor1_Model…
#> 3  0.00310 0.00599   roc_auc binary     0.959    30 0.00290 Preprocessor1_Model…
#> 4 32       0.000464  roc_auc binary     0.927    30 0.00484 Preprocessor1_Model…
#> 5  1       0.0000359 roc_auc binary     0.873    30 0.00533 Preprocessor1_Model…
```
:::




## Out-of-sample predictions

If we used `save_pred = TRUE` to keep the out-of-sample predictions for each resample during tuning, we can obtain those predictions, along with the tuning parameters and resample identifier, using `collect_predictions()`:




::: {.cell layout-align="center"}

```{.r .cell-code}
collect_predictions(recipe_res)
#> # A tibble: 38,740 × 8
#>    .pred_bad .pred_good id           .row     cost   rbf_sigma Class .config    
#>        <dbl>      <dbl> <chr>       <int>    <dbl>       <dbl> <fct> <chr>      
#>  1     0.333      0.667 Bootstrap01     1 0.000977 0.000000215 good  Preprocess…
#>  2     0.333      0.667 Bootstrap01     9 0.000977 0.000000215 good  Preprocess…
#>  3     0.333      0.667 Bootstrap01    10 0.000977 0.000000215 bad   Preprocess…
#>  4     0.333      0.667 Bootstrap01    12 0.000977 0.000000215 bad   Preprocess…
#>  5     0.333      0.667 Bootstrap01    14 0.000977 0.000000215 bad   Preprocess…
#>  6     0.333      0.667 Bootstrap01    15 0.000977 0.000000215 good  Preprocess…
#>  7     0.333      0.667 Bootstrap01    16 0.000977 0.000000215 bad   Preprocess…
#>  8     0.333      0.667 Bootstrap01    22 0.000977 0.000000215 bad   Preprocess…
#>  9     0.333      0.667 Bootstrap01    23 0.000977 0.000000215 good  Preprocess…
#> 10     0.333      0.667 Bootstrap01    24 0.000977 0.000000215 bad   Preprocess…
#> # ℹ 38,730 more rows
```
:::




We can obtain the hold-out sets for all the resamples augmented with the predictions using `augment()`, which provides opportunities for flexible visualization of model results:




::: {.cell layout-align="center"}

```{.r .cell-code}
augment(recipe_res) %>%
  ggplot(aes(V3, .pred_good, color = Class)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~Class)
```

::: {.cell-output-display}
![](figs/augment-preds-1.svg){fig-align='center' width=672}
:::
:::




## Session information {#session-info}




::: {.cell layout-align="center"}

```
#> ─ Session info ─────────────────────────────────────────────────────
#>  setting  value
#>  version  R version 4.4.2 (2024-10-31)
#>  os       macOS Sequoia 15.3.1
#>  system   aarch64, darwin20
#>  ui       X11
#>  language (EN)
#>  collate  en_US.UTF-8
#>  ctype    en_US.UTF-8
#>  tz       America/Los_Angeles
#>  date     2025-03-07
#>  pandoc   3.6.1 @ /usr/local/bin/ (via rmarkdown)
#>  quarto   1.6.42 @ /Applications/quarto/bin/quarto
#> 
#> ─ Packages ─────────────────────────────────────────────────────────
#>  package    * version date (UTC) lib source
#>  broom      * 1.0.7   2024-09-26 [1] CRAN (R 4.4.1)
#>  dials      * 1.4.0   2025-02-13 [1] CRAN (R 4.4.2)
#>  dplyr      * 1.1.4   2023-11-17 [1] CRAN (R 4.4.0)
#>  ggplot2    * 3.5.1   2024-04-23 [1] CRAN (R 4.4.0)
#>  infer      * 1.0.7   2024-03-25 [1] CRAN (R 4.4.0)
#>  kernlab    * 0.9-33  2024-08-13 [1] CRAN (R 4.4.0)
#>  mlbench    * 2.1-6   2024-12-30 [1] CRAN (R 4.4.1)
#>  parsnip    * 1.3.0   2025-02-14 [1] CRAN (R 4.4.2)
#>  purrr      * 1.0.4   2025-02-05 [1] CRAN (R 4.4.1)
#>  recipes    * 1.1.1   2025-02-12 [1] CRAN (R 4.4.1)
#>  rlang        1.1.5   2025-01-17 [1] CRAN (R 4.4.2)
#>  rsample    * 1.2.1   2024-03-25 [1] CRAN (R 4.4.0)
#>  tibble     * 3.2.1   2023-03-20 [1] CRAN (R 4.4.0)
#>  tidymodels * 1.3.0   2025-02-21 [1] CRAN (R 4.4.1)
#>  tune       * 1.3.0   2025-02-21 [1] CRAN (R 4.4.1)
#>  workflows  * 1.2.0   2025-02-19 [1] CRAN (R 4.4.1)
#>  yardstick  * 1.3.2   2025-01-22 [1] CRAN (R 4.4.1)
#> 
#>  [1] /Users/emilhvitfeldt/Library/R/arm64/4.4/library
#>  [2] /Library/Frameworks/R.framework/Versions/4.4-arm64/Resources/library
#>  * ── Packages attached to the search path.
#> 
#> ────────────────────────────────────────────────────────────────────
```
:::
