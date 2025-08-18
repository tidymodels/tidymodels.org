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

To use code in this article,  you will need to install the following packages: filtro.

You can construct new scoring objects using `class_score()`. This article is a guide to creating new scoring objects. 

## Scoring object

The `class_score` is a parent class. There are a few properties to this object:

::: {.cell layout-align="center"}

```{.r .cell-code}
library(filtro)
args(class_score)
#> function (outcome_type = c("numeric", "factor"), predictor_type = c("numeric", 
#> "factor"), case_weights = logical(0), range = integer(0), inclusive = logical(0), 
#>     fallback_value = integer(0), score_type = character(0), sorts = function() NULL, 
#>     direction = character(0), deterministic = logical(0), tuning = logical(0), 
#>     calculating_fn = function() NULL, label = character(0), packages = character(0), 
#>     results = data.frame()) 
#> NULL
```
:::

-   `outcome_type`: What types of outcome can the method handle?

-   `predictor_type`: What types of predictor can the method handle?

-   `case_weights`: Does the method accpet case weights? 

-   `range`: Are there known ranges for the statistic?

-   `inclusive`: Are these ranges inclusive at the bounds?

-   `fallback_value`: What is a value that can be used for the statistic so that it will never be eliminated?

-   `score_type`: What is the column name that will be used for the statistic values?

-   (Not used) `sorts`: How should the values be sorted (from most- to least-important)?

-   `direction`: What direction of values indicates the most important values?

-   `deterministic`: Does the fitting process use random numbers?

-   `tuning`: Does the method have tuning parameters?

-   `calculating_fn`: What function is used to estimate the values from data?

-   `label`: What label to use when printing?

-   `packages`: What packages, if any, are required to train the method?

-   `results`: A slot for the results once the method is fitted.

## Scoring object for specific to filter method

The `class_score_aov` is a subclass of `class_score`. This subclass allows additional properties to be introduced: 

::: {.cell layout-align="center"}

```{.r .cell-code}
class_score_aov <- S7::new_class(
  "class_score_aov",
  parent = class_score,
  properties = list(
    # Represent the score as -log10(p_value)?
    neg_log10 = S7::new_property(S7::class_logical, default = TRUE)
  )
)
```
:::

`score_aov_pval` is an instance (i.e., object) of the `class_score_aov` subclass, created using its constructor function: 

::: {.cell layout-align="center"}

```{.r .cell-code}
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

`score_aov_fstat` is another instance (i.e., object) of the `class_score_aov` subclass:  

::: {.cell layout-align="center"}

```{.r .cell-code}
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
