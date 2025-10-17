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

::: {.cell layout-align="center"}

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

::: {.cell layout-align="center"}

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

::: {.cell layout-align="center"}

```{.r .cell-code}
svm_mod <-
  svm_rbf(mode = "classification", cost = tune(), rbf_sigma = tune()) %>%
  set_engine("kernlab")
```
:::

These two objects (the recipe and model) will be combined into a single object via the `workflow()` function from the [workflows](https://workflows.tidymodels.org/) package; this object will be used in the optimization process. 

::: {.cell layout-align="center"}

```{.r .cell-code}
svm_wflow <-
  workflow() %>%
  add_model(svm_mod) %>%
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
#> maximum number of iterations reached 0.02644788 -4.678954e-05maximum number of iterations reached 4.883479e-05 -1.410116e-11maximum number of iterations reached 0.0006198503 -7.83456e-07maximum number of iterations reached 0.02616499 -4.815298e-05maximum number of iterations reached 4.783692e-05 -3.574141e-12maximum number of iterations reached 0.000532615 -6.369733e-07maximum number of iterations reached 0.02563573 -3.389226e-05maximum number of iterations reached 4.597583e-05 -1.520256e-11maximum number of iterations reached 0.0009844145 -1.428013e-06
#> ✓ Initialization complete
#> 
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> maximum number of iterations reached 0.001012865 -8.406313e-09
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> maximum number of iterations reached 0.001302281 -7.618055e-09
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> maximum number of iterations reached 0.0005753808 1.382829e-09
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
#> i Fold10: preprocessor 1/1, model 1/1
#> i Fold10: preprocessor 1/1, model 1/1 (predictions)
#> ✓ Estimating performance
#> i Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i Estimating performance
#> i Fold01: preprocessor 1/1
#> i Fold01: preprocessor 1/1, model 1/1
#> i Fold01: preprocessor 1/1, model 1/1 (predictions)
#> i Fold02: preprocessor 1/1
#> i Fold02: preprocessor 1/1, model 1/1
#> i Fold02: preprocessor 1/1, model 1/1 (predictions)
#> i Fold03: preprocessor 1/1
#> i Fold03: preprocessor 1/1, model 1/1
#> i Fold03: preprocessor 1/1, model 1/1 (predictions)
#> i Fold04: preprocessor 1/1
#> i Fold04: preprocessor 1/1, model 1/1
#> i Fold04: preprocessor 1/1, model 1/1 (predictions)
#> i Fold05: preprocessor 1/1
#> i Fold05: preprocessor 1/1, model 1/1
#> i Fold05: preprocessor 1/1, model 1/1 (predictions)
#> i Fold06: preprocessor 1/1
#> i Fold06: preprocessor 1/1, model 1/1
#> i Fold06: preprocessor 1/1, model 1/1 (predictions)
#> i Fold07: preprocessor 1/1
#> i Fold07: preprocessor 1/1, model 1/1
#> i Fold07: preprocessor 1/1, model 1/1 (predictions)
#> i Fold08: preprocessor 1/1
#> i Fold08: preprocessor 1/1, model 1/1
#> i Fold08: preprocessor 1/1, model 1/1 (predictions)
#> i Fold09: preprocessor 1/1
#> i Fold09: preprocessor 1/1, model 1/1
#> i Fold09: preprocessor 1/1, model 1/1 (predictions)
#> i Fold10: preprocessor 1/1
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
  collect_metrics(search_res) %>% 
  arrange(.iter)

estimates
#> # A tibble: 55 × 10
#>         cost   rbf_sigma num_comp .metric .estimator  mean     n std_err .config
#>        <dbl>       <dbl>    <int> <chr>   <chr>      <dbl> <int>   <dbl> <chr>  
#>  1  0.177       1   e-10        0 roc_auc binary     0.357    10  0.112  pre1_m…
#>  2  0.000977    3.16e- 3        5 roc_auc binary     0.346    10  0.114  pre2_m…
#>  3  2.38        1   e+ 0       10 roc_auc binary     0.821    10  0.0141 pre3_m…
#>  4 32           3.16e- 8       15 roc_auc binary     0.349    10  0.114  pre4_m…
#>  5  0.0131      1   e- 5       20 roc_auc binary     0.347    10  0.116  pre5_m…
#>  6 29.1         9.86e- 1        5 roc_auc binary     0.798    10  0.0144 iter01 
#>  7  0.00154     9.93e- 1       17 roc_auc binary     0.394    10  0.0819 iter02 
#>  8  5.26        9.99e- 1        8 roc_auc binary     0.833    10  0.0139 iter03 
#>  9  6.46        1.62e- 1        2 roc_auc binary     0.794    10  0.0104 iter04 
#> 10  8.73        8.67e- 1       12 roc_auc binary     0.804    10  0.0172 iter05 
#> # ℹ 45 more rows
#> # ℹ 1 more variable: .iter <int>
```
:::

The best performance of the initial set of candidate values was `AUC = 0.8214598 `. The best results were achieved at iteration 25 with a corresponding AUC value of 0.9008576. The five best results are:

::: {.cell layout-align="center"}

```{.r .cell-code}
show_best(search_res, metric = "roc_auc")
#> # A tibble: 5 × 10
#>    cost rbf_sigma num_comp .metric .estimator  mean     n std_err .config .iter
#>   <dbl>     <dbl>    <int> <chr>   <chr>      <dbl> <int>   <dbl> <chr>   <int>
#> 1  2.13    0.0389        9 roc_auc binary     0.901    10 0.00930 iter25     25
#> 2  1.80    0.0402        9 roc_auc binary     0.901    10 0.00935 iter23     23
#> 3  4.59    0.0253        9 roc_auc binary     0.901    10 0.00935 iter43     43
#> 4  3.42    0.0366        9 roc_auc binary     0.901    10 0.00944 iter26     26
#> 5  2.05    0.0520        9 roc_auc binary     0.901    10 0.00940 iter17     17
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
#>  version  R version 4.5.1 (2025-06-13)
#>  language (EN)
#>  date     2025-10-17
#>  pandoc   3.6.3
#>  quarto   1.8.25
#> 
#> ─ Packages ─────────────────────────────────────────────────────────
#>  package      version date (UTC) source
#>  broom        1.0.9   2025-07-28 CRAN (R 4.5.0)
#>  dials        1.4.2   2025-09-04 CRAN (R 4.5.0)
#>  dplyr        1.1.4   2023-11-17 CRAN (R 4.5.0)
#>  ggplot2      4.0.0   2025-09-11 CRAN (R 4.5.0)
#>  infer        1.0.9   2025-06-26 CRAN (R 4.5.0)
#>  kernlab      0.9-33  2024-08-13 CRAN (R 4.5.0)
#>  modeldata    1.5.1   2025-08-22 CRAN (R 4.5.0)
#>  parsnip      1.3.3   2025-08-31 CRAN (R 4.5.0)
#>  purrr        1.1.0   2025-07-10 CRAN (R 4.5.0)
#>  recipes      1.3.1   2025-05-21 CRAN (R 4.5.0)
#>  rlang        1.1.6   2025-04-11 CRAN (R 4.5.0)
#>  rsample      1.3.1   2025-07-29 CRAN (R 4.5.0)
#>  themis       1.0.3   2025-01-23 CRAN (R 4.5.0)
#>  tibble       3.3.0   2025-06-08 CRAN (R 4.5.0)
#>  tidymodels   1.4.1   2025-09-08 CRAN (R 4.5.0)
#>  tune         2.0.0   2025-09-01 CRAN (R 4.5.0)
#>  workflows    1.3.0   2025-08-27 CRAN (R 4.5.0)
#>  yardstick    1.3.2   2025-01-22 CRAN (R 4.5.0)
#> 
#> ────────────────────────────────────────────────────────────────────
```
:::

 
