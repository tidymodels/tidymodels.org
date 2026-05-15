---
title: "Iterative Bayesian optimization of a classification model"
categories:
  - tuning
  - SVMs
type: learn-subsection
weight: 3
description: | 
  Identify the best hyperparameters for a model using Bayesian optimization of iterative search.
toc: true
toc-depth: 2
r-packages:
  - tidymodels
  - kernlab
  - themis
  - future
include-after-body: ../../../html/resources.html
---

  

## Introduction

To use code in this article,  you will need to install the following packages: future, kernlab, themis, and tidymodels.

Many of the examples for model tuning focus on [grid search](/learn/work/tune-svm/). For that method, all the candidate tuning parameter combinations are defined prior to evaluation. Alternatively, _iterative search_ can be used to analyze the existing tuning parameter results and then _predict_ which tuning parameters to try next. 

There are a variety of methods for iterative search and the focus in this article is on _Bayesian optimization_. For more information on this method, these resources might be helpful:

* [_Practical bayesian optimization of machine learning algorithms_](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C7&q=Practical+Bayesian+Optimization+of+Machine+Learning+Algorithms&btnG=) (2012). J Snoek, H Larochelle, and RP Adams. Advances in neural information.  

* [_A Tutorial on Bayesian Optimization for Machine Learning_](https://www.cs.toronto.edu/~rgrosse/courses/csc411_f18/tutorials/tut8_adams_slides.pdf) (2018). R Adams.

 * [_Gaussian Processes for Machine Learning_](http://www.gaussianprocess.org/gpml/) (2006). C E Rasmussen and C Williams.

* [Other articles!](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C7&q="Bayesian+Optimization"&btnG=)

## Cell segmenting revisited

To demonstrate this approach to tuning models, let's return to the cell segmentation data from the [Getting Started](/start/resampling/) article on resampling: 

::: {.cell layout-align="center"}

```{.r .cell-code}
library(tidymodels)
library(modeldata)

# Load data
data(cells)

set.seed(2369)
tr_te_split <- initial_split(cells |> select(-case), prop = 3/4)
cell_train <- training(tr_te_split)
cell_test  <- testing(tr_te_split)

set.seed(1697)
folds <- vfold_cv(cell_train, v = 10)
```
:::

## The tuning scheme

Since the predictors are highly correlated, we can used a recipe to convert the original predictors to principal component scores. There is also slight class imbalance in these data; about 64% of the data are poorly segmented. To mitigate this, the data will be down-sampled at the end of the pre-processing so that the number of poorly and well segmented cells occur with equal frequency. We can use a recipe for all this pre-processing, but the number of principal components will need to be _tuned_ so that we have enough (but not too many) representations of the data. 

::: {.cell layout-align="center"}

```{.r .cell-code}
library(themis)

cell_pre_proc <-
  recipe(class ~ ., data = cell_train) |>
  step_YeoJohnson(all_predictors()) |>
  step_normalize(all_predictors()) |>
  step_pca(all_predictors(), num_comp = tune()) |>
  step_downsample(class)
```
:::

In this analysis, we will use a support vector machine to model the data. Let's use a radial basis function (RBF) kernel and tune its main parameter ($\sigma$). Additionally, the main SVM parameter, the cost value, also needs optimization. 

::: {.cell layout-align="center"}

```{.r .cell-code}
svm_mod <-
  svm_rbf(mode = "classification", cost = tune(), rbf_sigma = tune()) |>
  set_engine("kernlab")
```
:::

These two objects (the recipe and model) will be combined into a single object via the `workflow()` function from the [workflows](https://workflows.tidymodels.org/) package; this object will be used in the optimization process. 

::: {.cell layout-align="center"}

```{.r .cell-code}
svm_wflow <-
  workflow() |>
  add_model(svm_mod) |>
  add_recipe(cell_pre_proc)
```
:::

From this object, we can derive information about what parameters are slated to be tuned. A parameter set is derived by: 

::: {.cell layout-align="center"}

```{.r .cell-code}
svm_set <- extract_parameter_set_dials(svm_wflow)
svm_set
#> Collection of 3 parameters for tuning
#> 
#>  identifier      type    object
#>        cost      cost nparam[+]
#>   rbf_sigma rbf_sigma nparam[+]
#>    num_comp  num_comp nparam[+]
#> 
```
:::

The default range for the number of PCA components is rather small for this data set. A member of the parameter set can be modified using the `update()` function. Let's constrain the search to one to twenty components by updating the `num_comp` parameter. Additionally, the lower bound of this parameter is set to zero which specifies that the original predictor set should also be evaluated (i.e., with no PCA step at all): 

::: {.cell layout-align="center"}

```{.r .cell-code}
svm_set <- 
  svm_set |> 
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
  svm_wflow |> 
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
#> maximum number of iterations reached 0.02644788 -4.678954e-05maximum number of iterations reached 4.883479e-05 -1.405376e-11maximum number of iterations reached 0.0006198503 -7.83456e-07maximum number of iterations reached 0.02616499 -4.815298e-05maximum number of iterations reached 4.783692e-05 -3.531675e-12maximum number of iterations reached 0.000532615 -6.369734e-07maximum number of iterations reached 0.02563573 -3.389226e-05maximum number of iterations reached 4.597584e-05 -1.510886e-11maximum number of iterations reached 0.0009844145 -1.428013e-06
#> ✓ Initialization complete
#> 
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> maximum number of iterations reached 0.01591622 -1.060394e-06maximum number of iterations reached 0.01572637 -9.970078e-07
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> maximum number of iterations reached 0.01522807 -7.939643e-07
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1 (prediction data)
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1 (prediction data)
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1 (prediction data)
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1 (prediction data)
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1 (prediction data)
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1 (prediction data)
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1 (prediction data)
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1 (prediction data)
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1 (prediction data)
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1 (prediction data)
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
```
:::

The resulting tibble is a stacked set of rows of the rsample object with an additional column for the iteration number:

::: {.cell layout-align="center"}

```{.r .cell-code}
search_res
#> # Tuning results
#> # 10-fold cross-validation 
#> # A tibble: 510 × 5
#>    splits             id     .metrics         .notes           .iter
#>    <list>             <chr>  <list>           <list>           <int>
#>  1 <split [1362/152]> Fold01 <tibble [5 × 7]> <tibble [0 × 4]>     0
#>  2 <split [1362/152]> Fold02 <tibble [5 × 7]> <tibble [0 × 4]>     0
#>  3 <split [1362/152]> Fold03 <tibble [5 × 7]> <tibble [0 × 4]>     0
#>  4 <split [1362/152]> Fold04 <tibble [5 × 7]> <tibble [0 × 4]>     0
#>  5 <split [1363/151]> Fold05 <tibble [5 × 7]> <tibble [0 × 4]>     0
#>  6 <split [1363/151]> Fold06 <tibble [5 × 7]> <tibble [0 × 4]>     0
#>  7 <split [1363/151]> Fold07 <tibble [5 × 7]> <tibble [0 × 4]>     0
#>  8 <split [1363/151]> Fold08 <tibble [5 × 7]> <tibble [0 × 4]>     0
#>  9 <split [1363/151]> Fold09 <tibble [5 × 7]> <tibble [0 × 4]>     0
#> 10 <split [1363/151]> Fold10 <tibble [5 × 7]> <tibble [0 × 4]>     0
#> # ℹ 500 more rows
```
:::

As with grid search, we can summarize the results over resamples:

::: {.cell layout-align="center"}

```{.r .cell-code}
estimates <- 
  collect_metrics(search_res) |> 
  arrange(.iter)

estimates
#> # A tibble: 55 × 10
#>         cost rbf_sigma num_comp .metric .estimator  mean     n std_err .config  
#>        <dbl>     <dbl>    <int> <chr>   <chr>      <dbl> <int>   <dbl> <chr>    
#>  1  0.177     1   e-10        0 roc_auc binary     0.357    10  0.112  pre1_mod…
#>  2  0.000977  3.16e- 3        5 roc_auc binary     0.346    10  0.114  pre2_mod…
#>  3  2.38      1   e+ 0       10 roc_auc binary     0.821    10  0.0141 pre3_mod…
#>  4 32         3.16e- 8       15 roc_auc binary     0.349    10  0.114  pre4_mod…
#>  5  0.0131    1   e- 5       20 roc_auc binary     0.347    10  0.116  pre5_mod…
#>  6 31.2       1.61e- 5        0 roc_auc binary     0.877    10  0.0122 iter01   
#>  7  0.00105   3.49e-10       10 roc_auc binary     0.238    10  0.0585 iter02   
#>  8 24.2       4.41e- 2        1 roc_auc binary     0.770    10  0.0109 iter03   
#>  9  3.31      1.06e- 6        6 roc_auc binary     0.347    10  0.115  iter04   
#> 10 20.6       4.71e- 1        0 roc_auc binary     0.814    10  0.0114 iter05   
#> # ℹ 45 more rows
#> # ℹ 1 more variable: .iter <int>
```
:::

The best performance of the initial set of candidate values was `AUC = 0.8214598 `. The best results were achieved at iteration 38 with a corresponding AUC value of 0.9006475. The five best results are:

::: {.cell layout-align="center"}

```{.r .cell-code}
show_best(search_res, metric = "roc_auc")
#> # A tibble: 5 × 10
#>    cost rbf_sigma num_comp .metric .estimator  mean     n std_err .config .iter
#>   <dbl>     <dbl>    <int> <chr>   <chr>      <dbl> <int>   <dbl> <chr>   <int>
#> 1  4.70    0.0288        9 roc_auc binary     0.901    10 0.00926 iter38     38
#> 2  4.93    0.0310        9 roc_auc binary     0.900    10 0.00933 iter48     48
#> 3  5.09    0.0309        9 roc_auc binary     0.900    10 0.00934 iter30     30
#> 4  5.28    0.0245        8 roc_auc binary     0.899    10 0.00982 iter45     45
#> 5  4.80    0.0385        8 roc_auc binary     0.898    10 0.00995 iter43     43
```
:::

A plot of the search iterations can be created via:

::: {.cell layout-align="center"}

```{.r .cell-code}
autoplot(search_res, type = "performance")
```

::: {.cell-output-display}
![](figs/bo-plot-1.svg){fig-align='center' width=672}
:::
:::

There are many parameter combinations have roughly equivalent results. 

How did the parameters change over iterations? 

::: {.cell layout-align="center"}

```{.r .cell-code}
autoplot(search_res, type = "parameters") + 
  labs(x = "Iterations", y = NULL)
```

::: {.cell-output-display}
![](figs/bo-param-plot-1.svg){fig-align='center' width=864}
:::
:::

## Session information {#session-info}

::: {.cell layout-align="center"}

```
#> ─ Session info ─────────────────────────────────────────────────────
#>  version  R version 4.6.0 (2026-04-24)
#>  language (EN)
#>  pandoc   3.1.3
#>  quarto   1.9.37
#> 
#> ─ Packages ─────────────────────────────────────────────────────────
#>  package      version date (UTC)
#>  broom        1.0.13  2026-05-14
#>  dials        1.4.3   2026-04-11
#>  dplyr        1.2.1   2026-04-03
#>  future       1.70.0  2026-03-14
#>  ggplot2      4.0.3   2026-04-22
#>  infer        1.1.0   2025-12-18
#>  kernlab      0.9-33  2024-08-13
#>  parsnip      1.6.0   2026-05-14
#>  purrr        1.2.2   2026-04-10
#>  recipes      1.3.2   2026-04-02
#>  rlang        1.2.0   2026-04-06
#>  rsample      1.3.2   2026-01-30
#>  themis       1.0.3   2025-01-23
#>  tibble       3.3.1   2026-01-11
#>  tidymodels   1.5.0   2026-04-23
#>  tune         2.1.0   2026-04-17
#>  workflows    1.3.0   2025-08-27
#>  yardstick    1.4.0   2026-04-07
#> 
#> ────────────────────────────────────────────────────────────────────
```
:::

 
