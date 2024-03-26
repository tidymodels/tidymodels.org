---
title: "An introduction to calibration with tidymodels"
categories:
  - probably
  - yardstick
  - classification
  - calibration
type: learn-subsection
weight: 5
description: | 
  Learn how the probably package can improve classification and regression models.
toc: true
toc-depth: 2
include-after-body: ../../../resources.html
---








To use code in this article,  you will need to install the following packages: discrim, klaR, probably, and tidymodels. The probably package should be version 1.0.0 or greater.

There are essentially three different parts to a predictive model: 

 - the pre-processing stage (e.g., feature engineering, normalization, etc.)
 - model fitting (actually training the model)
 - post-processing (such as optimizing a probability threshold)

This article demonstrates a post-processing tool called model calibration. After the model fit, we might be able to improve a model by altering the predicted values.  

A classification model is well-calibrated if its probability estimate is consistent with the rate that the event occurs "in the wild." If you are not familiar with calibration, there are references at the end of this article.

To get started, load some packages: 


::: {.cell layout-align="center"}

```{.r .cell-code}
library(tidymodels)
library(probably)
library(discrim)

tidymodels_prefer()
theme_set(theme_bw())
options(pillar.advice = FALSE, pillar.min_title_chars = Inf)
```
:::



## An example: predicting cell segmentation quality

The modeldata package contains a data set called `cells`. Initially distributed by [Hill and Haney (2007)](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-340), they showed how to create models that predict the _quality_ of the image analysis of cells. The outcome has two levels `"PS"` (for poorly segmented images) or `"WS"` (well-segmented). There are 56 image features that can be used to build a classifier. 

Let's load the data, remove an unwanted column, and look at the outcome frequencies: 


::: {.cell layout-align="center"}

```{.r .cell-code}
data(cells)
cells$case <- NULL

dim(cells)
#> [1] 2019   57
cells %>% count(class)
#> # A tibble: 2 × 2
#>   class     n
#>   <fct> <int>
#> 1 PS     1300
#> 2 WS      719
```
:::


There is a class imbalance but that will not affect our work here. 

Let's make a 75% to 25% split of the data into training and testing using `initial_split()`. We'll also create a set of 10-fold cross-validation indices for model resampling. 


::: {.cell layout-align="center"}

```{.r .cell-code}
set.seed(8928)
split <- initial_split(cells, strata = class)
cells_tr <- training(split)
cells_te <- testing(split)

cells_rs <- vfold_cv(cells_tr, strata = class)
```
:::


Now that there are data to be modeled, let's get to it!

## A naive Bayes model

We'll show the utility of calibration tools by using a type of model that, in this instance, is likely to produce a poorly calibrated model. The naive Bayes classifier is a well-established model that assumes that the predictors are statistically _independent_ of one another (to simplify the calculations).  While that is certainly not the case for these data, the model can be effective at discriminating between the classes. Unfortunately, when there are many predictors in the model, it has a tendency to produce class probability distributions that are pathological. The predictions tend to gravitate to values near zero or one, producing distributions that are "U"-shaped ([Kuhn and Johnson, 2013](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C7&q=%22Applied+Predictive+Modeling%22&btnG=)). 

To demonstrate, let's set up the model:


::: {.cell layout-align="center"}

```{.r .cell-code}
bayes_wflow <-
  workflow() %>%
  add_formula(class ~ .) %>%
  add_model(naive_Bayes())
```
:::


We'll resample the model first so that we can get a good assessment of the results. During the resampling process, two metrics are used to judge how well the model worked. First, the area under the ROC curve is used to measure the ability of the model to separate the classes (using probability predictions). Second, the Brier score can measure how close the probability estimates are to the actual outcome values (zero or one). The `collect_metrics()` function shows the resampling estimates: 


::: {.cell layout-align="center"}

```{.r .cell-code}
cls_met <- metric_set(roc_auc, brier_class)
# We'll save the out-of-sample predictions to visualize them. 
ctrl <- control_resamples(save_pred = TRUE)

bayes_res <-
  bayes_wflow %>%
  fit_resamples(cells_rs, metrics = cls_met, control = ctrl)

collect_metrics(bayes_res)
#> # A tibble: 2 × 6
#>   .metric     .estimator  mean     n std_err .config             
#>   <chr>       <chr>      <dbl> <int>   <dbl> <chr>               
#> 1 brier_class binary     0.201    10 0.0101  Preprocessor1_Model1
#> 2 roc_auc     binary     0.855    10 0.00919 Preprocessor1_Model1
```
:::


The ROC score is impressive! However, the Brier value indicates that the probability values, while discriminating well, are not very realistic. A value of 0.25 is the "bad model" threshold when there are two classes (a value of zero being the best possible result). 

### But is it calibrated? 

Spoilers: no. It is not. 

The first clue is the extremely U-shaped distribution of the probability scores (facetted by the true class value): 


::: {.cell layout-align="center"}

```{.r .cell-code}
collect_predictions(bayes_res) %>%
  ggplot(aes(.pred_PS)) +
  geom_histogram(col = "white", bins = 40) +
  facet_wrap(~ class, ncol = 1) +
  geom_rug(col = "blue", alpha = 1 / 2) + 
  labs(x = "Probability Estimate of PS")
```

::: {.cell-output-display}
![](figs/prob-hist-1.svg){fig-align='center' width=60%}
:::
:::


There are almost no cells with moderate probability estimates. Furthermore, when the model is incorrect, it is "confidently incorrect". 

The probably package has tools for visualizing and correcting models with poor calibration properties. 

The most common plot is to break the predictions into about ten equally sized buckets and compute the actual event rate within each. For example, if a bin captures the samples predicted to be poorly segmented with probabilities between 20% and 30%, we should expect about a 25% event rate (i.e., the bin midpoint) within that partition. Here's a plot with ten bins: 


::: {.cell layout-align="center"}

```{.r .cell-code}
cal_plot_breaks(bayes_res)
```

::: {.cell-output-display}
![](figs/break-plot-1.svg){fig-align='center' width=60%}
:::
:::


The probabilities are not showing very good accuracy. 

There is also a similar function that can use moving windows with overlapping partitions. This provides a little more detail: 


::: {.cell layout-align="center"}

```{.r .cell-code}
cal_plot_windowed(bayes_res, step_size = 0.025)
```

::: {.cell-output-display}
![](figs/break-windowed-1.svg){fig-align='center' width=60%}
:::
:::


Bad. Still bad. 

Finally, for two class outcomes, we can fit a logistic generalized additive model (GAM) and examine the trend. 


::: {.cell layout-align="center"}

```{.r .cell-code}
cal_plot_logistic(bayes_res)
```

::: {.cell-output-display}
![](figs/break-logistic-1.svg){fig-align='center' width=60%}
:::
:::


Ooof. 

## Remediation

The good news is that we can do something about this. There are tools to "fix" the probability estimates so that they have better properties, such as falling along the diagonal lines in the diagnostic plots shown above. Different methods improve the predictions in different ways. 

The most common approach is the fit a logistic regression model to the data (with the probability estimates as the predictor). The probability predictions from this model are then used as the calibrated estimate. By default, a generalized additive model is used for this fit, but the `smooth = FALSE` argument can use simple linear effects. 

If effect, the GAM model estimates the probability regions where the model is off (as shown in the diagnostic plot). For example, suppose that when the model predicts a 2% event rate, the GAM model estimates that it under-predicts the probability by 5% (relative to the observed data). Given this gap, new predictions are adjusted up so that the probability estimates are more in-line with the data.  

How do we know if this works? There are a set of `cal_validate_*()` functions that can use holdout data to resample the model with and without the calibration tool of choice. Since we already resampled the model, we'll use those results to estimate 10 more logistic regressions and use the out-of-sample data to estimate performance. 

`collect_metrics()` can again be used to see the performance statistics. We'll also use `cal_plot_windowed()` on the calibrated holdout data to get a visual assessment:  


::: {.cell layout-align="center"}

```{.r .cell-code}
logit_val <- cal_validate_logistic(bayes_res, metrics = cls_met, save_pred = TRUE)
collect_metrics(logit_val)
#> # A tibble: 4 × 7
#>   .metric     .type        .estimator  mean     n std_err .config
#>   <chr>       <chr>        <chr>      <dbl> <int>   <dbl> <chr>  
#> 1 brier_class uncalibrated binary     0.201    10 0.0101  config 
#> 2 roc_auc     uncalibrated binary     0.855    10 0.00919 config 
#> 3 brier_class calibrated   binary     0.154    10 0.00612 config 
#> 4 roc_auc     calibrated   binary     0.854    10 0.00959 config

collect_predictions(logit_val) %>%
  filter(.type == "calibrated") %>%
  cal_plot_windowed(truth = class, estimate = .pred_PS, step_size = 0.025) +
  ggtitle("Logistic calibration via GAM")
```

::: {.cell-output-display}
![](figs/logistic-cal-1.svg){fig-align='center' width=60%}
:::
:::


That's a lot better but it is problematic that the calibrated predictions do not reach zero or one. 

A different approach is to use isotonic regression. This method can result in very few unique probability estimates. The probably package has a version of isotonic regression that resamples the process to produce more unique probabilities: 


::: {.cell layout-align="center"}

```{.r .cell-code}
set.seed(1212)
iso_val <- cal_validate_isotonic_boot(bayes_res, metrics = cls_met, 
                                      save_pred = TRUE, times = 25)
collect_metrics(iso_val)
#> # A tibble: 4 × 7
#>   .metric     .type        .estimator  mean     n std_err .config
#>   <chr>       <chr>        <chr>      <dbl> <int>   <dbl> <chr>  
#> 1 brier_class uncalibrated binary     0.201    10 0.0101  config 
#> 2 roc_auc     uncalibrated binary     0.855    10 0.00919 config 
#> 3 brier_class calibrated   binary     0.150    10 0.00500 config 
#> 4 roc_auc     calibrated   binary     0.855    10 0.00917 config

collect_predictions(iso_val) %>%
  filter(.type == "calibrated") %>%
  cal_plot_windowed(truth = class, estimate = .pred_PS, step_size = 0.025) +
  ggtitle("Isotonic regression calibration")
```

::: {.cell-output-display}
![](figs/isoreg-cal-1.svg){fig-align='center' width=60%}
:::
:::


Much better. However, there is a slight bias since the estimated points are consistently above the identity line on the 45-degree angle. 

Finally, we can also test out [Beta calibration](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C7&q=%22Beyond+sigmoids%22+calibration&btnG=): 


::: {.cell layout-align="center"}

```{.r .cell-code}
beta_val <- cal_validate_beta(bayes_res, metrics = cls_met, save_pred = TRUE)
collect_metrics(beta_val)
#> # A tibble: 4 × 7
#>   .metric     .type        .estimator  mean     n std_err .config
#>   <chr>       <chr>        <chr>      <dbl> <int>   <dbl> <chr>  
#> 1 brier_class uncalibrated binary     0.201    10 0.0101  config 
#> 2 roc_auc     uncalibrated binary     0.855    10 0.00919 config 
#> 3 brier_class calibrated   binary     0.145    10 0.00427 config 
#> 4 roc_auc     calibrated   binary     0.855    10 0.00919 config

collect_predictions(beta_val) %>%
  filter(.type == "calibrated") %>%
  cal_plot_windowed(truth = class, estimate = .pred_PS, step_size = 0.025) +
  ggtitle("Beta calibration")
```

::: {.cell-output-display}
![](figs/beta-cal-1.svg){fig-align='center' width=60%}
:::
:::


Also a big improvement but it does poorly at the lower end of the scale. 

Beta calibration appears to have the best results. We'll save a model that is trained using all of the out-of-sample predictions from the original naive Bayes resampling results. 

We can also fit the final naive Bayes model to predict the test set: 


::: {.cell layout-align="center"}

```{.r .cell-code}
cell_cal <- cal_estimate_beta(bayes_res)
bayes_fit <- bayes_wflow %>% fit(data = cells_tr)
```
:::


The `cell_cal` object can be used to enact the calibration for new predictions (as we'll see in a minute).

## Test set results

First, we make our ordinary predictions: 


::: {.cell layout-align="center"}

```{.r .cell-code}
cell_test_pred <- augment(bayes_fit, new_data = cells_te)
cell_test_pred %>% cls_met(class, .pred_PS)
#> # A tibble: 2 × 3
#>   .metric     .estimator .estimate
#>   <chr>       <chr>          <dbl>
#> 1 roc_auc     binary         0.840
#> 2 brier_class binary         0.226
```
:::


These metric estimates are very consistent with the resampled performance estimates. 

We can then use our `cell_cal` object with the `cal_apply()` function:


::: {.cell layout-align="center"}

```{.r .cell-code}
cell_test_cal_pred <-
  cell_test_pred %>%
  cal_apply(cell_cal)
cell_test_cal_pred %>% dplyr::select(class, starts_with(".pred_"))
#> # A tibble: 505 × 4
#>    class .pred_class .pred_PS .pred_WS
#>    <fct> <fct>          <dbl>    <dbl>
#>  1 PS    PS            0.882    0.118 
#>  2 WS    WS            0.215    0.785 
#>  3 WS    WS            0.0768   0.923 
#>  4 PS    PS            0.833    0.167 
#>  5 PS    PS            0.947    0.0534
#>  6 WS    WS            0.210    0.790 
#>  7 PS    PS            0.852    0.148 
#>  8 PS    PS            0.724    0.276 
#>  9 WS    WS            0.341    0.659 
#> 10 WS    PS            0.602    0.398 
#> # ℹ 495 more rows
```
:::


Note that `cal_apply()` recomputed the hard class predictions in the `.pred_class` column. It is possible that the changes in the probability estimates could invalidate the original hard class estimates. 

What do the calibrated test set results show? 


::: {.cell layout-align="center"}

```{.r .cell-code}
cell_test_cal_pred %>% cls_met(class, .pred_PS)
#> # A tibble: 2 × 3
#>   .metric     .estimator .estimate
#>   <chr>       <chr>          <dbl>
#> 1 roc_auc     binary         0.840
#> 2 brier_class binary         0.153
cell_test_cal_pred %>%
  cal_plot_windowed(truth = class, estimate = .pred_PS, step_size = 0.025)
```

::: {.cell-output-display}
![](figs/calibrated-res-1.svg){fig-align='center' width=60%}
:::
:::


Much better. The test set results also agree with the results from `cal_validate_beta().` 

## Other model types

probably can also calibrate classification models with more than two outcome levels. The functions `cal_*_multinomial()` use a multinomial model in the same spirit as the logistic regression model. Isotonic and Beta calibration can also be used via a "one versus all" approach that builds a set of binary calibrators and normalizes their results at the end (to ensure that they add to one). 

For regression models, there is `cal_plot_regression()` and `cal_*_linear()`. The latter uses `lm()` or `mgcv::gam()` to create a calibrator object. 
 
## Some background references


 - Kull, Meelis, Telmo M. Silva Filho, and Peter Flach. "[Beyond sigmoids: How to obtain well-calibrated probabilities from binary classifiers with beta calibration.](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C7&q=%22Beyond+sigmoids%22+calibration&btnG=)" (2017): 5052-5080

- Niculescu-Mizil, Alexandru, and Rich Caruana. "[Predicting good probabilities with supervised learning](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C7&q=%E2%80%9CPredicting+Good+Probabilities+with+Supervised+Learning%E2%80%9D&btnG=)." In _Proceedings of the 22nd international conference on Machine learning_, pp. 625-632. 2005.


## Session information


::: {.cell layout-align="center"}

```
#> ─ Session info ─────────────────────────────────────────────────────
#>  setting  value
#>  version  R version 4.3.3 (2024-02-29)
#>  os       macOS Sonoma 14.4.1
#>  system   aarch64, darwin20
#>  ui       X11
#>  language (EN)
#>  collate  en_US.UTF-8
#>  ctype    en_US.UTF-8
#>  tz       America/Los_Angeles
#>  date     2024-03-26
#>  pandoc   2.17.1.1 @ /opt/homebrew/bin/ (via rmarkdown)
#> 
#> ─ Packages ─────────────────────────────────────────────────────────
#>  package    * version date (UTC) lib source
#>  broom      * 1.0.5   2023-06-09 [1] CRAN (R 4.3.0)
#>  dials      * 1.2.1   2024-02-22 [1] CRAN (R 4.3.1)
#>  discrim    * 1.0.1   2023-03-08 [1] CRAN (R 4.3.0)
#>  dplyr      * 1.1.4   2023-11-17 [1] CRAN (R 4.3.1)
#>  ggplot2    * 3.5.0   2024-02-23 [1] CRAN (R 4.3.1)
#>  infer      * 1.0.7   2024-03-25 [1] CRAN (R 4.3.1)
#>  klaR       * 1.7-3   2023-12-13 [1] CRAN (R 4.3.1)
#>  parsnip    * 1.2.1   2024-03-22 [1] CRAN (R 4.3.1)
#>  probably   * 1.0.3   2024-02-23 [1] CRAN (R 4.3.1)
#>  purrr      * 1.0.2   2023-08-10 [1] CRAN (R 4.3.0)
#>  recipes    * 1.0.10  2024-02-18 [1] CRAN (R 4.3.1)
#>  rlang        1.1.3   2024-01-10 [1] CRAN (R 4.3.1)
#>  rsample    * 1.2.1   2024-03-25 [1] CRAN (R 4.3.1)
#>  tibble     * 3.2.1   2023-03-20 [1] CRAN (R 4.3.0)
#>  tidymodels * 1.2.0   2024-03-25 [1] CRAN (R 4.3.1)
#>  tune       * 1.2.0   2024-03-20 [1] CRAN (R 4.3.1)
#>  workflows  * 1.1.4   2024-02-19 [1] CRAN (R 4.3.1)
#>  yardstick  * 1.3.1   2024-03-21 [1] CRAN (R 4.3.1)
#> 
#>  [1] /Users/emilhvitfeldt/Library/R/arm64/4.3/library
#>  [2] /Library/Frameworks/R.framework/Versions/4.3-arm64/Resources/library
#> 
#> ────────────────────────────────────────────────────────────────────
```
:::
