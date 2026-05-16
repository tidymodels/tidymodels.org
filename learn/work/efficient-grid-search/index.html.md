---
title: "Efficient Grid Search"
categories:
  - tuning
  - preprocessing
  - postprocessing
type: learn-subsection
weight: 2
description: |
  A discussion on the techniques that makes `tune_grid()` fast.
toc: true
toc-depth: 2
r-packages:
  - tidymodels
  - ggforce
  - patchwork
include-after-body: ../../../html/resources.html
---

::: {.cell}

:::

## Introduction

This article describes strategies to make grid search more efficient. We are documenting the process that makes `tune_grid()` work and the tricks that we use to make it fast. There's also a theme of *grid search can be really fast if you do it right* and that's not by accident.

There is a parallel discussion (pun intended) of these topics in [*Tidy Models with R*](https://www.tmwr.org/grid-search#efficient-grids). This article has a little more detail about the mechanics of the process.

We won't talk about [racing](https://aml4td.org/chapters/grid-search.html#sec-racing) in this article; that might be the most efficient way to screen a grid, but it is an algorithm in itself and benefits from everything that we'll discuss here.

If you landed on this page because you are wondering how to make grid search run fast for your data set, [racing in combination with parallel processing](https://www.tmwr.org/workflow-sets#racing-example) is probably the fastest way to do it.

## The Overall Grid Tuning Procedure {#sec-overview}

tidymodels has three possible components to add to a model pipeline (a.k.a. a *workflow*):

- Preprocessors: computations that prepare data for the model
- Supervised model: the actual model fit
- Postprocessing: adjustments to model predictions

Each of these stages can have tuning parameters. Some examples:

- We might add a spline basis expansion to a predictor to enable to have a nonlinear trend in the model. We take one predictor and create multiple columns based on it. As we add new spline columns, the model can be more complex. We don't know how many columns to add, so we try different values and see which maximizes performance.

- If many predictors are correlated with one another, we could determine which predictors (and how many) can be removed to reduce multicollinearity. One approach is to have a threshold for the maximum allowable pairwise correlations and have an algorithm remove the smallest set of predictors to satisfy the constraint. The threshold would need to be tuned.

- Most models have tuning parameters. Boosted trees have many parameters, but two of the main ones are the learning rate and the number of trees in the ensemble. Both usually require tuning.

- For binary classification models, especially those with a class imbalance, it is possible that the default 50% probability threshold that is used to define what is "an event" does not satisfy the user's needs. For example, when screening blood bank samples, we want a model to err on the side of rejecting disease-free donations while minimizing the number of truly diseased samples that are misclassified as disease-free. In other words, we might want to tune the probability threshold to achieve a high specificity metric.

Once we know the tuning parameters and which pipeline phases they are from, we can create a grid of values to evaluate. Since a pipeline can have multiple tuning parameters, a *candidate* is defined as a single set of parameter values that could be tested. A set of multiple candidate values for tuning parameters is a grid.

For grid search, we evaluate each candidate in the grid by training the entire pipeline, predicting a set of data, and computing performance metrics that estimate the candidate’s efficacy. Once this is complete, we can select the candidate set that offers the best value.

This page outlines the general process and describes different constraints and approaches to efficiently evaluating a grid of candidate values. Unexpectedly, the tools we use to increase efficiency often lead to increasingly complex routines for computing performance on the grid.

We’ll break down some of the complications and exploitations that are relevant for model tuning (via grid search). We’ll also start with very simple use cases and steadily increase the tasks' and the solutions' complexity.

## Scenario A: Two Hyperparameters

In this case, let's start with a hypothetical classification example where our outcome class has two levels: event and nonevent.

For illustration, let's also suppose that there are a large number of potential predictors (say thousands). We might want to include a dimensionality reduction tool to make it easier for the supervised model. One such preprocessor is the Uniform Manifold Approximation and Projection ([UMAP](https://aml4td.org/chapters/embeddings.html#sec-umap)) method. This is an advanced multidimensional scaling method that can take a large number of predictors and reduce them down to a handful of potentially informative features. This method has a fair number of tuning parameters, and it is pretty sensitive to their values.

As an example, we've chosen to optimize UMAP's number of nearest neighbors. This can significantly affect the complexity of the dimension reduction. To keep things simple, we will investigate two values: a single nearest neighbor and 30 nearest neighbors (a pretty wide range). In tidymodels, the argument name for this parameter is `neighbors`.

Once we have our preprocessed data, we'll add it to a boosted tree model, such as xgboost. This also has many parameters, and again, we will focus on a single one to keep our discussions simple. Tree-based models commonly have tuning parameters that restrict how deep trees can become during training (to avoid overfitting). A common tool for this is to allow further splits only if the current data set has at least $n_{min}$ values. The argument name here is `min_n,` and we'll consider three possible values: 1, 50, and 100.

When people talk about grid search, they tend to focus on all possible combinations of our candidate parameter values (we'll discuss other grids below). In our case, we have a $2\times 3 = 6$ candidate grid. In tidymodels:

::: {.cell}

```{.r .cell-code}
library(tidymodels)
parameters(neighbors(c(1, 30)), min_n(c(1, 100))) |>
  grid_regular(levels = c(2, 3)) |>
  arrange(neighbors, min_n)
```

::: {.cell-output .cell-output-stdout}

```
# A tibble: 6 × 2
  neighbors min_n
      <int> <int>
1         1     1
2         1    50
3         1   100
4        30     1
5        30    50
6        30   100
```

:::
:::

Visually:

::: {.cell layout-align="center"}
::: {.cell-output-display}
![](index_files/figure-html/grid-a-1.png){fig-align='center' width=50%}
:::
:::

To tune our model, we would iterate over this grid, fitting six different models and evaluating their performance to choose the best one.

There's a problem, though: UMAP is a very expensive preprocessing method. If we loop through the rows of our grid, we'll end up estimating UMAP times even though there are only two possible values. That's incredibly inefficient.

### Problem: DRY (Don't Repeat Yourself)

If we have an inexpensive preprocessor for our data, repeating it is not a big problem; we would probably just keep things simple and effectively use a `for` loop over our rows.

However, there is a much better solution to make the computations as inexpensive as possible.

### Solution: Stagewise Estimation and Conditional Execution

We can separate our hyperparameters into three possible sets for the three possible stages of the pipeline: preprocessing, the supervised model, and postprocessing. In our example, we have a single tuning parameter each for preprocessing and the supervised model.

Stagewise execution is the process of incrementally fitting our pipeline for each of the three possible stages. We first "fit" the preprocessor (UMAP), then give the UMAP results to our boosted tree.

If we had multiple preprocessor tuning parameters, we would initially train the first candidate set of preprocessing parameters.

Since we are breaking out the pipeline into three stages, we can conditionally execute those calculations. This means we can find the unique combinations of preprocessing parameters and loop over them initially. For example, let's visualize our grid as a table where each block represents a value of the tuning parameter:

::: {.cell layout-align="center"}
::: {.cell-output-display}
![](index_files/figure-html/block-grid-1-1.png){fig-align='center' width=50%}
:::
:::

The purple blocks represent our UMAP preprocessing parameter (`neighbors`). The six rows in the grid represent the six pipeline candidates that we need to evaluate.

We have two unique values, so we follow a process in which we fit the first UMAP condition (a single neighbor) and, conditional on it, fit the corresponding three boosted trees.

::: {.cell layout-align="center"}
::: {.cell-output-display}
![](index_files/figure-html/block-grid--1loop-1.png){fig-align='center' width=100%}
:::
:::

We end up training six different tree ensembles but estimate only two UMAP models. This strategy can save a lot of time, especially for regular grids.

This conditional execution strategy seems like a simple solution, and it is. However, it will become more complex if we recursively iterate over the model and post-processing parameters.

## Problem: Curse of Dimensionality

Part of the problem with the conventional wisdom that "grid search is massively inefficient" is the Curse of Dimensionality. As the problem dimensions increase, the volume of the search space becomes vast. In our case, this means that, as we add more tuning parameters, the parameter space explodes exponentially. For example, if we wanted to evaluate $d$ values of each of $p$ tuning parameters, we would need $d^p$ combinations. With five parameters that are evaluated on four distinct points, the grid would have 1,024 rows.

It's a difficult situation as long as you don't think about the problem very hard. There are two ways to avoid this issue:

- [Parallelize](http://tmwr.org/grid-search#parallel-processing) the tasks. If we can do 10 things at once, the cost of doing more drops significantly. Grid search is an "embarrassingly parallel" problem, and running the computations in parallel can produce speedups of several-fold magnitude. This is not hard to do in R or tidymodels.
- *You don't have to use regular grids*.

For the second point, let's talk about *Space-filling designs* ([SFD](https://aml4td.org/chapters/grid-search.html#sec-irregular-grid)).

## Solution: Space-Filling Designs

There is extensive literature on space-filling experimental designs. They are tools to make grids that try to do at least two things:

- Cover the $p$-dimensional tuning parameter space as much as possible.
- Ensure that each point is as far away from the others as possible (to avoid redundant tuning parameter combinations).

Let's look at a low-dimensional example with two parameters. We might use a regular $4\times 6$ grid, as shown on the left. It covers the space, and all points are equally far from one another, but it leaves significant gaps in the parameter space. It's effective but not efficient.

The plot on the right shows a *uniform* SFD with the same number of points. This also covers the space well and packs the points closer together, so there are fewer gaps. These grids also have the property that if I want a grid of size X, each parameter will have X unique values in the design. Unless a parameter has fewer than X possible values, we are exploring each parameter's effect more thoroughly.

::: {.cell layout-align="center"}
::: {.cell-output-display}
![](index_files/figure-html/sfd-1.png){fig-align='center' width=80%}
:::
:::

The best part of this is that SFDs can cover high-dimensional space very effectively. Say we have 10 tuning parameters, an SFD could cover this space with 50 points, and each parameter is sampled across 50 unique values.

## Scenario B: Submodels

“Submodels” are situations where a model can make predictions for different tuning parameter values *without* retraining. Two examples:

- *Boosting* ensembles take the same model and, using different case weights, create a sequential set of fits. For example, a tree ensemble created with 100 boosting iterations will contain 100 trees, each depending on the previous fits. For many implementations, you can fit the largest ensemble size (100 in this example) and, for free, make predictions on sizes 1 through 99.
- Some regularized models, such as glmnet, have a penalization parameter that attenuates the magnitude of the regression coefficients (think weight decay in neural networks). We often tune over this parameter. For glmnet, the model simultaneously creates a *path* of possible penalty values. With a single model fit, you can predict on *any* penalty values.

Not all models contain parameters that can use the “submodel trick.” For example, the random forest model is also an ensemble of trees. However, unlike boosting, those trees are not created sequentially; they are independent of one another.

However, when our model includes submodel parameters, there can be massive efficiency gains when conducting a grid search.

Let's add another parameter to our UMAP/boosting pipeline. The number of trees in the ensemble is a critical parameter. Since boosting is an iterative process, each tree depends on the trees before it. It's an important parameter, so let's add another parameter to our previous regular grid that evaluates 10 values for `trees`. Now our grid has $2\times 3 \times 10=60$ parameter combinations:

::: {.cell layout-align="center"}
::: {.cell-output-display}
![](index_files/figure-html/block-grid-2-1.png){fig-align='center' width=60%}
:::
:::

Boosting has to store each tree, which means we have everything we need to know what would have happened if we had stopped training after one, ten, or fifty trees. Many boosting implementations include an argument in their `predict()` methods to obtain predictions at any stage. tidymodels is aware of this and, when possible, it trains only the *minimal set* of supervised models.

Visually, here are the six models that are actually trained (note the configuration numbers on the y-axis):

::: {.cell layout-align="center"}
::: {.cell-output-display}
![](index_files/figure-html/block-grid-2-actual-1.png){fig-align='center' width=60%}
:::
:::

Let's expand our previous grid to cross it with 10 values for the number of trees in the ensemble. Now we have a grid with 60 rows (i.e., parameter candidates), but we only have to train six models, just as before. The submodel calculations are not free, but they are pretty close to free. If we tune over a large number of values for this parameter, we can efficiently achieve a significant increase in density in our grid.

### Problem: Unaligned Submodels

Recall that space-filling designs are very effective at uniformly filling the parameter space without an exponentially increasing number of grid points. Also, one design aspect is that each tuning parameter has unique values in the grid (i.e., not repeated values).

Overall, SFDs are, hands down, the most efficient designs for grid search. *However*, the approach can be very inefficient when there are submodels. Let's look at a scatterplot matrix of a SFD for our three parameters with 30 grid points:

::: {.cell layout-align="center"}
::: {.cell-output-display}
![](index_files/figure-html/unaligned-1.png){fig-align='center' width=70%}
:::
:::

If you look at the right-most columns, there are no values of `trees` that are repeated in the design. This, in effect, cancels the benefit of submodels. In this design, there are no combinations of `neighbors` or `min_n` that have more than one value of `trees`.

We could go back to a regular grid. That will be very efficient for the submodel parameters (assuming that there is one), but very inefficient for the other parameters.

Another option is a mixture of space-filling points and a regular grid.

### Solution: Mixed Grids

A mixed grid would create a space-filling design for the parameters that cannot exploit submodels and then, for each design point in that grid, create a dense sequence of values for the submodel parameter. This may significantly increase the number of grid points evaluated, but limits the number of models that are fit.

Here's an example with many more grid points (900), but trains on the same number of models as the 3D grid in the previous section (30).

::: {.cell layout-align="center"}
::: {.cell-output-display}
![](index_files/figure-html/mixed-1.png){fig-align='center' width=70%}
:::
:::

Recall that this is only really an issue when our model has submodel parameters and is inefficient to train. As a counterexample, the aforementioned glmnet model fits generalized linear models (e.g., linear or logistic regression) but includes a penalty parameter that can reduce model complexity, lowering the risk of overfitting. From a single model, it can quickly produce predictions for a large number of predictions across a sequence of penalty values. As with the tree example, an SFD would negate the submodel benefit, but the glmnet model trains very quickly, so an SFD would still be effective for tuning.

Currently, there are no tidymodels methods for making mixed grids. You would create one manaully using `grid_space_filling()` and tidyr's `crossing()`: 

```r
sfd_grid <- 
  parameters(neighbors(), min_n()) |> 
  grid_space_filling(size = 30, type = "uniform")

mixed_grid <- 
  sfd_grid |> 
  crossing(trees = value_seq(trees(), 30))
```

## Postprocessing

Postprocessing methods transform predictions to make them more useful. We can think of them as being either estimated or not. As was mentioned in the first section, if we have a classification model with two possible classes, we might want to adjust the rule that converts the predicted class probabilities to "event" or "no event". This enables us to optimize for what we care about (e.g., sensitivity or specificity). For this example, we are not estimating a parameter to test a potential threshold; we simply apply something like `ifelse(prob >= threshold, "event", "no event")` to transform the class predictions.

An example of an estimated postprocessor is calibration. Here, we use a model to estimate when we are underpredicting the outcome, and then make a correction to eliminate those errors. For example, in a regression model, if we are consistently underpredicting samples whose outcomes are at the high end of the distribution, we can estimate that error and add it when we make predictions in that range. It's actually more complicated than that, but this covers the ideas relevant for this discussion.

The most efficient way to execute these computations when we need to tune the postprocess is another case of conditional execution. For each combination of modeling and preprocessing parameters, we loop over the postprocessing tuning parameters to estimate (if needed) and apply the transformations.

The only real speed-up comes when the postprocessor is *not* estimated. In that case, there is a slight efficiency boost by storing all predictions in a data frame and applying the transformation to them all at once. For example, if we tuned a probability threshold over 50 values, we can row-bind them into a single data frame and use an efficient in-line application of `ifelse()`. It just saves us a `for` loop, but these small efficiencies can add up.
