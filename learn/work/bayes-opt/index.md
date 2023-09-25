---
title: "Iterative Bayesian optimization of a classification model"
categories:
  - model tuning
  - Bayesian optimization
  - SVMs
type: learn-subsection
weight: 3
description: | 
  Identify the best hyperparameters for a model using Bayesian optimization of iterative search.
toc: true
toc-depth: 2
include-after-body: ../../../resources.html
---







## Introduction

To use code in this article,  you will need to install the following packages: kernlab, modeldata, themis, and tidymodels.

Many of the examples for model tuning focus on [grid search](/learn/work/tune-svm/). For that method, all the candidate tuning parameter combinations are defined prior to evaluation. Alternatively, _iterative search_ can be used to analyze the existing tuning parameter results and then _predict_ which tuning parameters to try next. 

There are a variety of methods for iterative search and the focus in this article is on _Bayesian optimization_. For more information on this method, these resources might be helpful:

* [_Practical bayesian optimization of machine learning algorithms_](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C7&q=Practical+Bayesian+Optimization+of+Machine+Learning+Algorithms&btnG=) (2012). J Snoek, H Larochelle, and RP Adams. Advances in neural information.  

* [_A Tutorial on Bayesian Optimization for Machine Learning_](https://www.cs.toronto.edu/~rgrosse/courses/csc411_f18/tutorials/tut8_adams_slides.pdf) (2018). R Adams.

 * [_Gaussian Processes for Machine Learning_](http://www.gaussianprocess.org/gpml/) (2006). C E Rasmussen and C Williams.

* [Other articles!](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C7&q="Bayesian+Optimization"&btnG=)


## Cell segmenting revisited

To demonstrate this approach to tuning models, let's return to the cell segmentation data from the [Getting Started](/start/resampling/) article on resampling: 


::: {.cell layout-align="center" hash='cache/import-data_55799729effaf25afde25ffe183a1f89'}

```{.r .cell-code}
library(tidymodels)
library(modeldata)

# Load data
data(cells)

set.seed(2369)
tr_te_split <- initial_split(cells %>% select(-case), prop = 3/4)
cell_train <- training(tr_te_split)
cell_test  <- testing(tr_te_split)

set.seed(1697)
folds <- vfold_cv(cell_train, v = 10)
```
:::


## The tuning scheme

Since the predictors are highly correlated, we can used a recipe to convert the original predictors to principal component scores. There is also slight class imbalance in these data; about 64% of the data are poorly segmented. To mitigate this, the data will be down-sampled at the end of the pre-processing so that the number of poorly and well segmented cells occur with equal frequency. We can use a recipe for all this pre-processing, but the number of principal components will need to be _tuned_ so that we have enough (but not too many) representations of the data. 


::: {.cell layout-align="center" hash='cache/recipe_61d40f5037d0999e9cb27e53b4db2d21'}

```{.r .cell-code}
library(themis)

cell_pre_proc <-
  recipe(class ~ ., data = cell_train) %>%
  step_YeoJohnson(all_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), num_comp = tune()) %>%
  step_downsample(class)
```
:::


In this analysis, we will use a support vector machine to model the data. Let's use a radial basis function (RBF) kernel and tune its main parameter ($\sigma$). Additionally, the main SVM parameter, the cost value, also needs optimization. 


::: {.cell layout-align="center" hash='cache/model_c50db642bfcd5a983d8015c499484549'}

```{.r .cell-code}
svm_mod <-
  svm_rbf(mode = "classification", cost = tune(), rbf_sigma = tune()) %>%
  set_engine("kernlab")
```
:::


These two objects (the recipe and model) will be combined into a single object via the `workflow()` function from the [workflows](https://workflows.tidymodels.org/) package; this object will be used in the optimization process. 


::: {.cell layout-align="center" hash='cache/workflow_7973003642ea7eb6fd7dced472ed32ed'}

```{.r .cell-code}
svm_wflow <-
  workflow() %>%
  add_model(svm_mod) %>%
  add_recipe(cell_pre_proc)
```
:::


From this object, we can derive information about what parameters are slated to be tuned. A parameter set is derived by: 


::: {.cell layout-align="center" hash='cache/pset_f20780f808937594ded59c16540e05c5'}

```{.r .cell-code}
svm_set <- extract_parameter_set_dials(svm_wflow)
svm_set
#> Collection of 3 parameters for tuning
#> 
#>  identifier      type    object
#>        cost      cost nparam[+]
#>   rbf_sigma rbf_sigma nparam[+]
#>    num_comp  num_comp nparam[+]
```
:::


The default range for the number of PCA components is rather small for this data set. A member of the parameter set can be modified using the `update()` function. Let's constrain the search to one to twenty components by updating the `num_comp` parameter. Additionally, the lower bound of this parameter is set to zero which specifies that the original predictor set should also be evaluated (i.e., with no PCA step at all): 


::: {.cell layout-align="center" hash='cache/update_b880232f77e11836adb259277ffb556c'}

```{.r .cell-code}
svm_set <- 
  svm_set %>% 
  update(num_comp = num_comp(c(0L, 20L)))
```
:::


## Sequential tuning 

Bayesian optimization is a sequential method that uses a model to predict new candidate parameters for assessment. When scoring potential parameter value, the mean and variance of performance are predicted. The strategy used to define how these two statistical quantities are used is defined by an _acquisition function_. 

For example, one approach for scoring new candidates is to use a confidence bound. Suppose accuracy is being optimized. For a metric that we want to maximize, a lower confidence bound can be used. The multiplier on the standard error (denoted as $\kappa$) is a value that can be used to make trade-offs between **exploration** and **exploitation**. 

 * **Exploration** means that the search will consider candidates in untested space.

 * **Exploitation** focuses in areas where the previous best results occurred. 

The variance predicted by the Bayesian model is mostly spatial variation; the value will be large for candidate values that are not close to values that have already been evaluated. If the standard error multiplier is high, the search process will be more likely to avoid areas without candidate values in the vicinity. 

We'll use another acquisition function, _expected improvement_, that determines which candidates are likely to be helpful relative to the current best results. This is the default acquisition function. More information on these functions can be found in the [package vignette for acquisition functions](https://tune.tidymodels.org/articles/acquisition_functions.html). 


::: {.cell layout-align="center"}

```{.r .cell-code}
set.seed(12)
search_res <-
  svm_wflow %>% 
  tune_bayes(
    resamples = folds,
    # To use non-default parameter ranges
    param_info = svm_set,
    # Generate five at semi-random to start
    initial = 5,
    iter = 50,
    # How to measure performance?
    metrics = metric_set(roc_auc),
    control = control_bayes(no_improve = 30, verbose = TRUE)
  )
#> 
#> ❯  Generating a set of 5 initial parameter results
#> ✓ Initialization complete
#> 
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> ✓ Estimating performance
#> ! No improvement for 30 iterations; returning current results.
```
:::


The resulting tibble is a stacked set of rows of the rsample object with an additional column for the iteration number:


::: {.cell layout-align="center" hash='cache/show-iters_36dc3313d421a4f2e3388ef5a4773148'}

```{.r .cell-code}
search_res
#> # Tuning results
#> # 10-fold cross-validation 
#> # A tibble: 480 × 5
#>    splits             id     .metrics         .notes           .iter
#>    <list>             <chr>  <list>           <list>           <int>
#>  1 <split [1362/152]> Fold01 <tibble [5 × 7]> <tibble [0 × 3]>     0
#>  2 <split [1362/152]> Fold02 <tibble [5 × 7]> <tibble [0 × 3]>     0
#>  3 <split [1362/152]> Fold03 <tibble [5 × 7]> <tibble [0 × 3]>     0
#>  4 <split [1362/152]> Fold04 <tibble [5 × 7]> <tibble [0 × 3]>     0
#>  5 <split [1363/151]> Fold05 <tibble [5 × 7]> <tibble [0 × 3]>     0
#>  6 <split [1363/151]> Fold06 <tibble [5 × 7]> <tibble [0 × 3]>     0
#>  7 <split [1363/151]> Fold07 <tibble [5 × 7]> <tibble [0 × 3]>     0
#>  8 <split [1363/151]> Fold08 <tibble [5 × 7]> <tibble [0 × 3]>     0
#>  9 <split [1363/151]> Fold09 <tibble [5 × 7]> <tibble [0 × 3]>     0
#> 10 <split [1363/151]> Fold10 <tibble [5 × 7]> <tibble [0 × 3]>     0
#> # ℹ 470 more rows
```
:::


As with grid search, we can summarize the results over resamples:


::: {.cell layout-align="center" hash='cache/summarize-iters_658b86df928959f079fa6267607bb02e'}

```{.r .cell-code}
estimates <- 
  collect_metrics(search_res) %>% 
  arrange(.iter)

estimates
#> # A tibble: 52 × 10
#>        cost    rbf_sigma num_comp .metric .estimator  mean     n std_err .config
#>       <dbl>        <dbl>    <int> <chr>   <chr>      <dbl> <int>   <dbl> <chr>  
#>  1  0.00383      2.72e-6       17 roc_auc binary     0.348    10  0.114  Prepro…
#>  2  0.250        1.55e-2        7 roc_auc binary     0.879    10  0.0122 Prepro…
#>  3  0.0372       1.02e-9        3 roc_auc binary     0.242    10  0.0574 Prepro…
#>  4  1.28         8.13e-8        8 roc_auc binary     0.344    10  0.114  Prepro…
#>  5 10.3          1.37e-3       14 roc_auc binary     0.877    10  0.0117 Prepro…
#>  6 29.2          7.07e-1       17 roc_auc binary     0.788    10  0.0111 Iter1  
#>  7 30.4          8.70e-3       13 roc_auc binary     0.895    10  0.0101 Iter2  
#>  8  0.0374       4.25e-3       11 roc_auc binary     0.875    10  0.0123 Iter3  
#>  9 28.8          3.86e-3        4 roc_auc binary     0.874    10  0.0120 Iter4  
#> 10 21.5          7.38e-2       11 roc_auc binary     0.852    10  0.0115 Iter5  
#> # ℹ 42 more rows
#> # ℹ 1 more variable: .iter <int>
```
:::



The best performance of the initial set of candidate values was `AUC = 0.8793995 `. The best results were achieved at iteration 17 with a corresponding AUC value of 0.8995344. The five best results are:


::: {.cell layout-align="center" hash='cache/best_f8c6d844d88527be5cc8fd3bc15c5cef'}

```{.r .cell-code}
show_best(search_res, metric = "roc_auc")
#> # A tibble: 5 × 10
#>    cost rbf_sigma num_comp .metric .estimator  mean     n std_err .config .iter
#>   <dbl>     <dbl>    <int> <chr>   <chr>      <dbl> <int>   <dbl> <chr>   <int>
#> 1  27.3   0.00829        9 roc_auc binary     0.900    10 0.00996 Iter17     17
#> 2  29.3   0.00958       10 roc_auc binary     0.899    10 0.00959 Iter25     25
#> 3  25.1   0.0111         9 roc_auc binary     0.899    10 0.00996 Iter11     11
#> 4  31.1   0.00933       10 roc_auc binary     0.899    10 0.00968 Iter16     16
#> 5  27.6   0.00901        9 roc_auc binary     0.899    10 0.0100  Iter22     22
```
:::


A plot of the search iterations can be created via:


::: {.cell layout-align="center" hash='cache/bo-plot_6974f3b87e6101248a080765f5f9a452'}

```{.r .cell-code}
autoplot(search_res, type = "performance")
```

::: {.cell-output-display}
![](figs/bo-plot-1.svg){fig-align='center' width=672}
:::
:::


There are many parameter combinations have roughly equivalent results. 

How did the parameters change over iterations? 



::: {.cell layout-align="center" hash='cache/bo-param-plot_df3a97a65fc356046bdfa2a85e4de520'}

```{.r .cell-code}
autoplot(search_res, type = "parameters") + 
  labs(x = "Iterations", y = NULL)
```

::: {.cell-output-display}
![](figs/bo-param-plot-1.svg){fig-align='center' width=864}
:::
:::





## Session information {#session-info}


::: {.cell layout-align="center" hash='cache/si_5db2644d2f49a924bcfd72b2c3cad09a'}

```
#> ─ Session info ─────────────────────────────────────────────────────
#>  setting  value
#>  version  R version 4.3.1 (2023-06-16)
#>  os       macOS Ventura 13.5.2
#>  system   aarch64, darwin20
#>  ui       X11
#>  language (EN)
#>  collate  en_US.UTF-8
#>  ctype    en_US.UTF-8
#>  tz       America/Los_Angeles
#>  date     2023-09-25
#>  pandoc   3.1.1 @ /Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/ (via rmarkdown)
#> 
#> ─ Packages ─────────────────────────────────────────────────────────
#>  package    * version date (UTC) lib source
#>  broom      * 1.0.5   2023-06-09 [1] CRAN (R 4.3.0)
#>  dials      * 1.2.0   2023-04-03 [1] CRAN (R 4.3.0)
#>  dplyr      * 1.1.3   2023-09-03 [1] CRAN (R 4.3.0)
#>  ggplot2    * 3.4.3   2023-08-14 [1] CRAN (R 4.3.0)
#>  infer      * 1.0.5   2023-09-06 [1] CRAN (R 4.3.0)
#>  kernlab    * 0.9-32  2023-01-31 [1] CRAN (R 4.3.0)
#>  modeldata  * 1.2.0   2023-08-09 [1] CRAN (R 4.3.0)
#>  parsnip    * 1.1.1   2023-08-17 [1] CRAN (R 4.3.0)
#>  purrr      * 1.0.2   2023-08-10 [1] CRAN (R 4.3.0)
#>  recipes    * 1.0.8   2023-08-25 [1] CRAN (R 4.3.0)
#>  rlang      * 1.1.1   2023-04-28 [1] CRAN (R 4.3.0)
#>  rsample    * 1.2.0   2023-08-23 [1] CRAN (R 4.3.0)
#>  themis     * 1.0.2   2023-08-14 [1] CRAN (R 4.3.0)
#>  tibble     * 3.2.1   2023-03-20 [1] CRAN (R 4.3.0)
#>  tidymodels * 1.1.1   2023-08-24 [1] CRAN (R 4.3.0)
#>  tune       * 1.1.2   2023-08-23 [1] CRAN (R 4.3.0)
#>  workflows  * 1.1.3   2023-02-22 [1] CRAN (R 4.3.0)
#>  yardstick  * 1.2.0   2023-04-21 [1] CRAN (R 4.3.0)
#> 
#>  [1] /Users/emilhvitfeldt/Library/R/arm64/4.3/library
#>  [2] /Library/Frameworks/R.framework/Versions/4.3-arm64/Resources/library
#> 
#> ────────────────────────────────────────────────────────────────────
```
:::
