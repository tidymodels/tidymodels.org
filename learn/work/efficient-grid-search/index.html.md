---
title: "Efficient Grid Search"
categories:
  - tuning
type: learn-subsection
weight: 4
description: | 
  A general discussion on grids and how to efficiently estimate performance. .
toc: true
toc-depth: 2
include-after-body: ../../../resources.html
---






## Introduction

To use code in this article,  you will need to install the following packages: tidymodels.

This article demonstrates ....


## The Overall Grid Tuning Procedure

tidymodels has three phases of creating a model pipeline: 

 - preprocessors: computations that prepare data for the model
 - supervised model: the actual model fit
 - postprocessing: adjustments to model predictions 

Each of these stages can have tuning parameters. Some examples:

- We might add a spline basis expansion to a predictor to enable to have a nonlinear trend in the model. We take one predictor and create multiple columns that are based on the original column. As we add new spline columns, the model can be more complex. We don't know how many columns to add so we try different values and see which maximizes performance. 

- If many predictors are correlated with one another, we could determine which predictors (and how many) can be removed to reduce multicollinearity. One approach is the have a threshold for the maximum allowable pairwise correlations and have an algorithm remove the smallest set of predictors to satisfy the constraint. The threshold would need to be tuned. 

- Most models have tuning parameters. Boosted trees have many but two of the main parameters are the learning rate and the number of trees in the ensemble. Both usually require tuning. 

- For binary classification models, especially those with a class imbalance, it is possible that the default 50% probability threshold that is used to define what is "an event" does not satisfy the user's needs. For example, when screening blood bank samples, we want a model to err on the side of throwing out disease-free donations as long as we minimize the number of bad blood samples that truly are diseased. In other words, we might want to tune the probability threshold to achieve a high specificity metric. 

Once we know what the tuning parameters are and which phases of the pipeline they are from, we can create a grid of values to evaluate. Since a pipeline can have multiple tuning parameters, a _candidate_ is defined to be a single set of actual parameter values that could be tested. A set of multiple tuning parameter candidate values is a grid. 

For grid search, we evaluate each candidate in the grid by training the entire pipeline, predicting a set of data, and computing performance metrics that estimate the candidate’s efficacy. Once this is finished, we can pick the candidate set that shows the best value. 

This page outlines the general process and describes different constraints and approaches to efficiently evaluating a grid of candidate values. Unexpectedly, the tools that we use to increase efficiency often result in more and more complex routines to compute performance for the grid. 

We’ll break down some of the complications and exploitations that are relevant for model tuning (via grid search). We’ll also start with very simple use-cases and steadily increase the complexity of the task as well as the complexity of the solutions. 

## How Does Training Occur in the Pipeline?

We should describe how different models would accomplish the three stages of a model pipeline. 

On one hand, there are deep neural networks. These are highly nonlinear models whose model structure is defined as a sequence of layers. The architecture of the network is defined by different types of layers that are intended to do different things. The mantra of deep learning is to add layers to the network that can accomplish all of the tasks discussed above. There is little to no delineation regarding what is a pre- or post-processor; it is all part of the model and all parameters are estimated simultaneously. We will call this type of model a “simultaneous estimation” since it all happens at the same time. 

On the other hand is nearly every other type of model. In most cases, preprocessing tasks are separate estimation procedures that are executed independently of the model fit. There are cases such as principal component regression that conducts PCA signal extraction before passing the results to a routine that executes ordinary least squares. However, there are still two separate estimation tasks that are carried out in sequence, not simultaneously. We’ll call this type of training “stagewise estimation” since there are multiple, distinct training steps that occur in sequence. 

Our focus is 100% on stagewise estimation. 

## Stage-Wise Modeling and Conditional Execution

One important part of processing a grid of candidate values is to avoid repeating any computations wherever possible. This leads to the idea of conditional execution. Let’s think of an example. 

Suppose our pipeline consists of one preprocess and the model fit (i.e., no postprocessing). Let’s choose a fairly expensive preprocessing technique: UMAP feature extraction. UMAP has a variety of tuning parameters and one is the number of nearest neighbors to use when creating a network of nearby training set samples. Suppose that we will consider values ranging from 1 to 30. 

For our model, we’ll choose a random forest model. This also has tuning parameters and we’ll choose to optimize “m-try”; the number of predictors to randomly select each time a split is created in any decision tree. 

For illustration, let’s use a grid of points with six candidates: 



::: {.cell layout-align="center"}

```{.r .cell-code}
library(tidymodels)
umap_rf_param <- parameters(neighbors(c(1, 30)), mtry(c(1, 100)))
umap_rf_grid <- grid_regular(umap_rf_param, levels = c(2, 3))
umap_rf_grid %>% arrange(neighbors, mtry)
#> # A tibble: 6 × 2
#>   neighbors  mtry
#>       <int> <int>
#> 1         1     1
#> 2         1    50
#> 3         1   100
#> 4        30     1
#> 5        30    50
#> 6        30   100
```
:::



To evaluate these candidates, we could just loop through each row, train a pipeline with that row’s candidate values, predictor a holdout set, and then compute a performance statistic. What does “train a pipeline” mean though? For the first candidate, it means 

- Carry out the UMAP estimation process on the training set using a single nearest neighbor. 
- Apply UMAP to the training set and save the transformed data. 
- Estimate a random forest model with `mtry = 1` using the transformed training set. 

The first and third steps are two separate estimations.

The remaining tasks are to

- Apply UMAP to the holdout data. 
- Predict the holdout data with the fitted random forest model. 
- Compute the performance statistic of interest (e.g., R<sup>2</sup>, accuracy, RMSE, etc.)

For the second candidate, the process is exactly the same _except_ that the random forest model uses `mtry = 50` instead of `mtry = 1`. 

We _have_ to compute a new random forest model since it is a different model. However, the UMAP step is identical to the previous candidate’s preprocessor since it has the same values. Since UMAP is expensive, we end up spending a lot of time doing something that we have already done. 

This would not be the case if there were no connection between the preprocessing tuning parameters and the model parameters, for example, this grid would not have repeated computations: 



::: {.cell layout-align="center"}

```
#> # A tibble: 6 × 2
#>   neighbors  mtry
#>       <dbl> <int>
#> 1         5     1
#> 2        10     1
#> 3        15    50
#> 4        20    50
#> 5        25   100
#> 6        30   100
```
:::



In this case, the UMAP step would be different for each candidate and we would have to recompute it anyways. 

Once solution to avoiding redundant computations would be to cache the UMAP computations so that they could be reused later. 

A better solutions is _conditional execution_. In this case, we determine the unique set of candidate values associated with the preprocessor and loop over these. Within each loop, we we train and predict the random forest model across its candidate values. For the first grid of candidates contained in `umap_rf_grid`: 

:::: {.columns}

::: {.column width="10%"}

:::

::: {.column width="80%"}


```pseudocode
#| html-line-number: true
#| html-line-number-punc: ":"

\begin{algorithm}
\begin{algorithmic}
\State $\mathfrak{D}^{tr}$: training set
\State $\mathfrak{D}^{ho}$: holdout set
\For{$k \in \{1, 30\}$}
  \State Using $k$ neighbors, train UMAP on $\mathfrak{D}^{tr}$
  \State Apply UMAP to $\mathfrak{D}^{tr}$, creating $\widehat{\mathfrak{D}}^{tr}_k$
  \State Apply UMAP to $\mathfrak{D}^{ho}$, creating $\widehat{\mathfrak{D}}^{ho}_k$  
  \For{$m \in \{1, 50, 100\}$}
    \State Train a random forest model with $m_{try} = m$ on $\mathfrak{D}^{tr}_k$ to produce $\widehat{f}_{km}$
    \State Predict $\widehat{\mathfrak{D}}^{ho}_k$ with  $\widehat{f}_{km}$
    \State Compute performance statistic $\widehat{Q}_{km}$. 
  \EndFor
\EndFor
\State Determine the $k$ and $m$ calues corresponding to the best value of $\widehat{Q}_{km}$.
\end{algorithmic}
\end{algorithm}
```

:::

::: {.column width="10%"}

:::

::::


In this way, we evaluated six candidates via two UMAP models and six random forest models. We avoid four redundant and expensive UMAP fits.

We can organize this data using a nested structure:



::: {.cell layout-align="center"}

```{.r .cell-code}
umap_rf_nested_grid <- 
  umap_rf_grid %>% 
  group_nest(neighbors, .key = "second_stage")
umap_rf_nested_grid
#> # A tibble: 2 × 2
#>   neighbors       second_stage
#>       <int> <list<tibble[,1]>>
#> 1         1            [3 × 1]
#> 2        30            [3 × 1]
umap_rf_nested_grid$second_stage[[1]]
#> # A tibble: 3 × 1
#>    mtry
#>   <int>
#> 1     1
#> 2    50
#> 3   100
```
:::



In general, this is a good idea. Even when the second grid is used, there is no computational loss incurred by conditional execution. This is also true if the preprocessing technique is inexpensive. We will see one issue with conditional execution that comes up in section TODO when we can run the computations in parallel. 

## Sidebar: Types of Grids

Since the type of grid mattered for this example, let’s go on a quick “side quest” to talk about the two major types of grids. 

the effect of integers (max size etc) and also on duplicate computations

## Sidebar: Parallel Processing



## What are Submodels?


## How Can we Exploit Submodels? 


## Types of Postprocessors

Those needing tuning and those that are just applied. 



## Postprocessing Example: PRobability Threshold Optimization

