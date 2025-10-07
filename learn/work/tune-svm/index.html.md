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
#> maximum number of iterations reached 0.002729295 -0.002728731maximum number of iterations reached 1.48662e-05 -1.48662e-05maximum number of iterations reached 0.005087422 -0.005081074maximum number of iterations reached 0.003476184 -0.003475215maximum number of iterations reached 1.983896e-05 -1.983895e-05maximum number of iterations reached 0.006626915 -0.006616544maximum number of iterations reached 0.002515968 -0.002515484maximum number of iterations reached 1.34078e-05 -1.340779e-05maximum number of iterations reached 0.004555265 -0.0045506maximum number of iterations reached 0.002469115 -0.002468711maximum number of iterations reached 1.330233e-05 -1.330233e-05maximum number of iterations reached 0.00467122 -0.004667136maximum number of iterations reached 0.001989876 -0.001989645maximum number of iterations reached 1.315534e-05 -1.315534e-05maximum number of iterations reached 0.003656559 -0.003654453maximum number of iterations reached 0.002283848 -0.002283544maximum number of iterations reached 1.242395e-05 -1.242395e-05maximum number of iterations reached 0.004336801 -0.004333636maximum number of iterations reached 0.002963038 -0.002962428maximum number of iterations reached 1.496312e-05 -1.496312e-05maximum number of iterations reached 0.005434167 -0.005428241maximum number of iterations reached 0.002604825 -0.002604363maximum number of iterations reached 1.429271e-05 -1.429271e-05maximum number of iterations reached 0.004979161 -0.004973656maximum number of iterations reached 0.002666438 -0.002665922maximum number of iterations reached 1.370558e-05 -1.370558e-05maximum number of iterations reached 0.005090946 -0.005085138maximum number of iterations reached 0.001743169 -0.001743017maximum number of iterations reached 1.059289e-05 -1.059289e-05maximum number of iterations reached 0.00326213 -0.003260567maximum number of iterations reached 0.002568834 -0.002568403maximum number of iterations reached 1.354358e-05 -1.354357e-05maximum number of iterations reached 0.004861762 -0.004856653maximum number of iterations reached 0.002818794 -0.002818223maximum number of iterations reached 1.575347e-05 -1.575346e-05maximum number of iterations reached 0.005249384 -0.005243307maximum number of iterations reached 0.002860941 -0.002860314maximum number of iterations reached 1.503557e-05 -1.503557e-05maximum number of iterations reached 0.005400004 -0.005392986maximum number of iterations reached 0.003400418 -0.003399556maximum number of iterations reached 1.793883e-05 -1.793883e-05maximum number of iterations reached 0.006313314 -0.006304658maximum number of iterations reached 0.001877079 -0.001876912maximum number of iterations reached 1.127929e-05 -1.127928e-05maximum number of iterations reached 0.003670281 -0.003668525maximum number of iterations reached 0.002657858 -0.00265742maximum number of iterations reached 1.399915e-05 -1.399914e-05maximum number of iterations reached 0.005267624 -0.005262646maximum number of iterations reached 0.003063188 -0.003062394maximum number of iterations reached 1.569352e-05 -1.569351e-05maximum number of iterations reached 0.005752102 -0.005743002maximum number of iterations reached 0.00341126 -0.003410316maximum number of iterations reached 1.876099e-05 -1.876098e-05maximum number of iterations reached 0.00666891 -0.00665703maximum number of iterations reached 0.003056492 -0.003055746maximum number of iterations reached 1.558056e-05 -1.558056e-05maximum number of iterations reached 0.005753023 -0.005745269maximum number of iterations reached 0.002616596 -0.002616129maximum number of iterations reached 1.432853e-05 -1.432853e-05maximum number of iterations reached 0.004808037 -0.004803453maximum number of iterations reached 0.002710034 -0.002709487maximum number of iterations reached 1.425954e-05 -1.425954e-05maximum number of iterations reached 0.005062868 -0.005056853maximum number of iterations reached 0.002314831 -0.002314496maximum number of iterations reached 1.35017e-05 -1.35017e-05maximum number of iterations reached 0.004284309 -0.004281096maximum number of iterations reached 0.003593176 -0.003592102maximum number of iterations reached 1.881228e-05 -1.881227e-05maximum number of iterations reached 0.00656715 -0.006557095maximum number of iterations reached 0.002894065 -0.002893471maximum number of iterations reached 1.578179e-05 -1.578178e-05maximum number of iterations reached 0.005494275 -0.005487745maximum number of iterations reached 0.002421177 -0.002420827maximum number of iterations reached 1.39692e-05 -1.396919e-05maximum number of iterations reached 0.004434554 -0.004430894maximum number of iterations reached 0.002863577 -0.002863015maximum number of iterations reached 1.60938e-05 -1.60938e-05maximum number of iterations reached 0.005575677 -0.005569426maximum number of iterations reached 0.003103257 -0.003102474maximum number of iterations reached 1.578427e-05 -1.578427e-05maximum number of iterations reached 0.005594288 -0.005586607maximum number of iterations reached 0.003683707 -0.003682607maximum number of iterations reached 1.91626e-05 -1.916259e-05maximum number of iterations reached 0.006760405 -0.006751376maximum number of iterations reached 0.002493337 -0.002492961maximum number of iterations reached 1.428742e-05 -1.428742e-05maximum number of iterations reached 0.004858875 -0.004854543maximum number of iterations reached 0.002770911 -0.00277028maximum number of iterations reached 1.449988e-05 -1.449988e-05maximum number of iterations reached 0.005037426 -0.005031439
formula_res
#> # Tuning results
#> # Bootstrap sampling 
#> # A tibble: 30 × 5
#>    splits            id          .metrics          .notes           .predictions
#>    <list>            <chr>       <list>            <list>           <list>      
#>  1 <split [351/120]> Bootstrap01 <tibble [10 × 6]> <tibble [0 × 4]> <tibble>    
#>  2 <split [351/130]> Bootstrap02 <tibble [10 × 6]> <tibble [0 × 4]> <tibble>    
#>  3 <split [351/137]> Bootstrap03 <tibble [10 × 6]> <tibble [0 × 4]> <tibble>    
#>  4 <split [351/141]> Bootstrap04 <tibble [10 × 6]> <tibble [0 × 4]> <tibble>    
#>  5 <split [351/131]> Bootstrap05 <tibble [10 × 6]> <tibble [0 × 4]> <tibble>    
#>  6 <split [351/131]> Bootstrap06 <tibble [10 × 6]> <tibble [0 × 4]> <tibble>    
#>  7 <split [351/127]> Bootstrap07 <tibble [10 × 6]> <tibble [0 × 4]> <tibble>    
#>  8 <split [351/123]> Bootstrap08 <tibble [10 × 6]> <tibble [0 × 4]> <tibble>    
#>  9 <split [351/131]> Bootstrap09 <tibble [10 × 6]> <tibble [0 × 4]> <tibble>    
#> 10 <split [351/117]> Bootstrap10 <tibble [10 × 6]> <tibble [0 × 4]> <tibble>    
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
#>  1  0.000977 0.000000215   roc_auc binary         0.838 pre0_mod01_post0
#>  2  0.00310  0.00599       roc_auc binary         0.942 pre0_mod02_post0
#>  3  0.00984  0.0000000001  roc_auc binary         0.815 pre0_mod03_post0
#>  4  0.0312   0.00000278    roc_auc binary         0.832 pre0_mod04_post0
#>  5  0.0992   0.0774        roc_auc binary         0.968 pre0_mod05_post0
#>  6  0.315    0.00000000129 roc_auc binary         0.830 pre0_mod06_post0
#>  7  1        0.0000359     roc_auc binary         0.837 pre0_mod07_post0
#>  8  3.17     1             roc_auc binary         0.974 pre0_mod08_post0
#>  9 10.1      0.0000000167  roc_auc binary         0.832 pre0_mod09_post0
#> 10 32        0.000464      roc_auc binary         0.861 pre0_mod10_post0
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
#>  1  0.000977 0.000000215   roc_auc binary     0.871    30 0.00516 pre0_mod01_po…
#>  2  0.00310  0.00599       roc_auc binary     0.959    30 0.00290 pre0_mod02_po…
#>  3  0.00984  0.0000000001  roc_auc binary     0.822    30 0.00718 pre0_mod03_po…
#>  4  0.0312   0.00000278    roc_auc binary     0.871    30 0.00531 pre0_mod04_po…
#>  5  0.0992   0.0774        roc_auc binary     0.970    30 0.00261 pre0_mod05_po…
#>  6  0.315    0.00000000129 roc_auc binary     0.857    30 0.00624 pre0_mod06_po…
#>  7  1        0.0000359     roc_auc binary     0.873    30 0.00533 pre0_mod07_po…
#>  8  3.17     1             roc_auc binary     0.971    30 0.00248 pre0_mod08_po…
#>  9 10.1      0.0000000167  roc_auc binary     0.871    30 0.00534 pre0_mod09_po…
#> 10 32        0.000464      roc_auc binary     0.927    30 0.00484 pre0_mod10_po…
```
:::

The top combinations are:

::: {.cell layout-align="center"}

```{.r .cell-code}
show_best(formula_res, metric = "roc_auc")
#> # A tibble: 5 × 8
#>       cost rbf_sigma .metric .estimator  mean     n std_err .config         
#>      <dbl>     <dbl> <chr>   <chr>      <dbl> <int>   <dbl> <chr>           
#> 1  3.17    1         roc_auc binary     0.971    30 0.00248 pre0_mod08_post0
#> 2  0.0992  0.0774    roc_auc binary     0.970    30 0.00261 pre0_mod05_post0
#> 3  0.00310 0.00599   roc_auc binary     0.959    30 0.00290 pre0_mod02_post0
#> 4 32       0.000464  roc_auc binary     0.927    30 0.00484 pre0_mod10_post0
#> 5  1       0.0000359 roc_auc binary     0.873    30 0.00533 pre0_mod07_post0
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
#> maximum number of iterations reached 0.002742459 -0.0027419maximum number of iterations reached 1.41767e-05 -1.417669e-05maximum number of iterations reached 0.005063382 -0.005057462maximum number of iterations reached 0.003559933 -0.003558871maximum number of iterations reached 1.839635e-05 -1.839635e-05maximum number of iterations reached 0.006503999 -0.00649393maximum number of iterations reached 0.002514222 -0.002513769maximum number of iterations reached 1.337914e-05 -1.337914e-05maximum number of iterations reached 0.004701971 -0.004696892maximum number of iterations reached 0.002512296 -0.00251186maximum number of iterations reached 1.35322e-05 -1.353219e-05maximum number of iterations reached 0.004688318 -0.004684078maximum number of iterations reached 0.00201446 -0.002014239maximum number of iterations reached 1.181384e-05 -1.181384e-05maximum number of iterations reached 0.003806843 -0.003804562maximum number of iterations reached 0.002238874 -0.002238586maximum number of iterations reached 1.316689e-05 -1.316688e-05maximum number of iterations reached 0.004295401 -0.004292429maximum number of iterations reached 0.002887453 -0.002886869maximum number of iterations reached 1.481601e-05 -1.481601e-05maximum number of iterations reached 0.005540786 -0.005534606maximum number of iterations reached 0.002509744 -0.002509279maximum number of iterations reached 1.548108e-05 -1.548107e-05maximum number of iterations reached 0.004869151 -0.004863923maximum number of iterations reached 0.00271789 -0.002717372maximum number of iterations reached 1.344855e-05 -1.344854e-05maximum number of iterations reached 0.005088211 -0.005082189maximum number of iterations reached 0.001788083 -0.001787917maximum number of iterations reached 1.038728e-05 -1.038728e-05maximum number of iterations reached 0.00327505 -0.003273591maximum number of iterations reached 0.002565143 -0.002564693maximum number of iterations reached 1.340505e-05 -1.340505e-05maximum number of iterations reached 0.004797415 -0.004792541maximum number of iterations reached 0.002819414 -0.002818838maximum number of iterations reached 1.606027e-05 -1.606026e-05maximum number of iterations reached 0.005439585 -0.005432979maximum number of iterations reached 0.0027882 -0.002787594maximum number of iterations reached 1.524922e-05 -1.524922e-05maximum number of iterations reached 0.00519029 -0.005183748maximum number of iterations reached 0.003381619 -0.003380755maximum number of iterations reached 1.760863e-05 -1.760862e-05maximum number of iterations reached 0.006384695 -0.006376075maximum number of iterations reached 0.001945395 -0.001945211maximum number of iterations reached 1.065017e-05 -1.065016e-05maximum number of iterations reached 0.003457146 -0.003455547maximum number of iterations reached 0.002792796 -0.002792333maximum number of iterations reached 1.502743e-05 -1.502743e-05maximum number of iterations reached 0.005208387 -0.005203715maximum number of iterations reached 0.003050062 -0.00304923maximum number of iterations reached 1.520112e-05 -1.520112e-05maximum number of iterations reached 0.00570829 -0.005699808maximum number of iterations reached 0.003525424 -0.003524307maximum number of iterations reached 1.898649e-05 -1.898649e-05maximum number of iterations reached 0.00648104 -0.006469245maximum number of iterations reached 0.003044483 -0.003043757maximum number of iterations reached 1.531583e-05 -1.531583e-05maximum number of iterations reached 0.005595095 -0.005587695maximum number of iterations reached 0.002576566 -0.002576127maximum number of iterations reached 1.477244e-05 -1.477244e-05maximum number of iterations reached 0.004990352 -0.004985023maximum number of iterations reached 0.002763485 -0.002762901maximum number of iterations reached 1.446549e-05 -1.446549e-05maximum number of iterations reached 0.004964604 -0.004958588maximum number of iterations reached 0.002368979 -0.00236863maximum number of iterations reached 1.289772e-05 -1.289772e-05maximum number of iterations reached 0.004231297 -0.004228114maximum number of iterations reached 0.003641421 -0.003640323maximum number of iterations reached 1.969006e-05 -1.969005e-05maximum number of iterations reached 0.00653699 -0.006527311maximum number of iterations reached 0.002840029 -0.002839469maximum number of iterations reached 1.690192e-05 -1.690192e-05maximum number of iterations reached 0.005462389 -0.005456maximum number of iterations reached 0.002396492 -0.002396135maximum number of iterations reached 1.380691e-05 -1.380691e-05maximum number of iterations reached 0.004663851 -0.004659728maximum number of iterations reached 0.002949473 -0.002948877maximum number of iterations reached 1.616891e-05 -1.61689e-05maximum number of iterations reached 0.005622439 -0.00561606maximum number of iterations reached 0.002911029 -0.002910376maximum number of iterations reached 1.621631e-05 -1.621631e-05maximum number of iterations reached 0.005602028 -0.005594715maximum number of iterations reached 0.003762223 -0.003761115maximum number of iterations reached 1.905902e-05 -1.905901e-05maximum number of iterations reached 0.006985212 -0.006975203maximum number of iterations reached 0.002524708 -0.002524315maximum number of iterations reached 1.500886e-05 -1.500886e-05maximum number of iterations reached 0.004750492 -0.004746452maximum number of iterations reached 0.002741921 -0.002741331maximum number of iterations reached 1.492422e-05 -1.492421e-05maximum number of iterations reached 0.005142 -0.005135487
recipe_res
#> # Tuning results
#> # Bootstrap sampling 
#> # A tibble: 30 × 5
#>    splits            id          .metrics          .notes           .predictions
#>    <list>            <chr>       <list>            <list>           <list>      
#>  1 <split [351/120]> Bootstrap01 <tibble [10 × 6]> <tibble [0 × 4]> <tibble>    
#>  2 <split [351/130]> Bootstrap02 <tibble [10 × 6]> <tibble [0 × 4]> <tibble>    
#>  3 <split [351/137]> Bootstrap03 <tibble [10 × 6]> <tibble [0 × 4]> <tibble>    
#>  4 <split [351/141]> Bootstrap04 <tibble [10 × 6]> <tibble [0 × 4]> <tibble>    
#>  5 <split [351/131]> Bootstrap05 <tibble [10 × 6]> <tibble [0 × 4]> <tibble>    
#>  6 <split [351/131]> Bootstrap06 <tibble [10 × 6]> <tibble [0 × 4]> <tibble>    
#>  7 <split [351/127]> Bootstrap07 <tibble [10 × 6]> <tibble [0 × 4]> <tibble>    
#>  8 <split [351/123]> Bootstrap08 <tibble [10 × 6]> <tibble [0 × 4]> <tibble>    
#>  9 <split [351/131]> Bootstrap09 <tibble [10 × 6]> <tibble [0 × 4]> <tibble>    
#> 10 <split [351/117]> Bootstrap10 <tibble [10 × 6]> <tibble [0 × 4]> <tibble>    
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
#> 1  3.17    1         roc_auc binary     0.971    30 0.00248 pre0_mod08_post0
#> 2  0.0992  0.0774    roc_auc binary     0.970    30 0.00261 pre0_mod05_post0
#> 3  0.00310 0.00599   roc_auc binary     0.959    30 0.00290 pre0_mod02_post0
#> 4 32       0.000464  roc_auc binary     0.927    30 0.00484 pre0_mod10_post0
#> 5  1       0.0000359 roc_auc binary     0.873    30 0.00533 pre0_mod07_post0
```
:::

## Out-of-sample predictions

If we used `save_pred = TRUE` to keep the out-of-sample predictions for each resample during tuning, we can obtain those predictions, along with the tuning parameters and resample identifier, using `collect_predictions()`:

::: {.cell layout-align="center"}

```{.r .cell-code}
collect_predictions(recipe_res)
#> # A tibble: 38,740 × 8
#>    .pred_bad .pred_good id          Class  .row     cost   rbf_sigma .config    
#>        <dbl>      <dbl> <chr>       <fct> <int>    <dbl>       <dbl> <chr>      
#>  1     0.333      0.667 Bootstrap01 good      1 0.000977 0.000000215 pre0_mod01…
#>  2     0.333      0.667 Bootstrap01 good      9 0.000977 0.000000215 pre0_mod01…
#>  3     0.333      0.667 Bootstrap01 bad      10 0.000977 0.000000215 pre0_mod01…
#>  4     0.333      0.667 Bootstrap01 bad      12 0.000977 0.000000215 pre0_mod01…
#>  5     0.333      0.667 Bootstrap01 bad      14 0.000977 0.000000215 pre0_mod01…
#>  6     0.333      0.667 Bootstrap01 good     15 0.000977 0.000000215 pre0_mod01…
#>  7     0.333      0.667 Bootstrap01 bad      16 0.000977 0.000000215 pre0_mod01…
#>  8     0.333      0.667 Bootstrap01 bad      22 0.000977 0.000000215 pre0_mod01…
#>  9     0.333      0.667 Bootstrap01 good     23 0.000977 0.000000215 pre0_mod01…
#> 10     0.333      0.667 Bootstrap01 bad      24 0.000977 0.000000215 pre0_mod01…
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
#>  version  R version 4.5.1 (2025-06-13)
#>  language (EN)
#>  date     2025-10-07
#>  pandoc   3.6.3
#>  quarto   1.8.25
#> 
#> ─ Packages ─────────────────────────────────────────────────────────
#>  package      version    date (UTC) source
#>  broom        1.0.9      2025-07-28 CRAN (R 4.5.0)
#>  dials        1.4.2      2025-09-04 CRAN (R 4.5.0)
#>  dplyr        1.1.4      2023-11-17 CRAN (R 4.5.0)
#>  ggplot2      4.0.0      2025-09-11 CRAN (R 4.5.0)
#>  infer        1.0.9      2025-06-26 CRAN (R 4.5.0)
#>  kernlab      0.9-33     2024-08-13 CRAN (R 4.5.0)
#>  mlbench      2.1-6      2024-12-30 CRAN (R 4.5.0)
#>  parsnip      1.3.3      2025-08-31 CRAN (R 4.5.0)
#>  purrr        1.1.0      2025-07-10 CRAN (R 4.5.0)
#>  recipes      1.3.1      2025-05-21 CRAN (R 4.5.0)
#>  rlang        1.1.6      2025-04-11 CRAN (R 4.5.0)
#>  rsample      1.3.1      2025-07-29 CRAN (R 4.5.0)
#>  tibble       3.3.0      2025-06-08 CRAN (R 4.5.0)
#>  tidymodels   1.4.1      2025-09-08 CRAN (R 4.5.0)
#>  tune         2.0.0.9000 2025-10-07 local
#>  workflows    1.3.0      2025-08-27 CRAN (R 4.5.0)
#>  yardstick    1.3.2      2025-01-22 CRAN (R 4.5.0)
#> 
#> ────────────────────────────────────────────────────────────────────
```
:::

