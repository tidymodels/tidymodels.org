---
title: "Hypothesis testing using resampling and tidy data"
categories:
  - statistical analysis
  - hypothesis testing
  - bootstrapping
type: learn-subsection
weight: 4
description: | 
  Perform common hypothesis tests for statistical inference using flexible functions.
toc: true
toc-depth: 2
include-after-body: ../../../resources.html
---







## Introduction

This article only requires the tidymodels package. 

The tidymodels package [infer](https://infer.tidymodels.org/) implements an expressive grammar to perform statistical inference that coheres with the `tidyverse` design framework. Rather than providing methods for specific statistical tests, this package consolidates the principles that are shared among common hypothesis tests into a set of 4 main verbs (functions), supplemented with many utilities to visualize and extract information from their outputs.

Regardless of which hypothesis test we're using, we're still asking the same kind of question: 

>Is the effect or difference in our observed data real, or due to chance? 

To answer this question, we start by assuming that the observed data came from some world where "nothing is going on" (i.e. the observed effect was simply due to random chance), and call this assumption our **null hypothesis**. (In reality, we might not believe in the null hypothesis at all; the null hypothesis is in opposition to the **alternate hypothesis**, which supposes that the effect present in the observed data is actually due to the fact that "something is going on.") We then calculate a **test statistic** from our data that describes the observed effect. We can use this test statistic to calculate a **p-value**, giving the probability that our observed data could come about if the null hypothesis was true. If this probability is below some pre-defined **significance level** $\alpha$, then we can reject our null hypothesis.

If you are new to hypothesis testing, take a look at 

* [Section 9.2 of _Statistical Inference via Data Science_](https://moderndive.com/9-hypothesis-testing.html#understanding-ht)
* The American Statistical Association's recent [statement on p-values](https://doi.org/10.1080/00031305.2016.1154108) 

The workflow of this package is designed around these ideas. Starting from some data set,

+ `specify()` allows you to specify the variable, or relationship between variables, that you're interested in,
+ `hypothesize()` allows you to declare the null hypothesis,
+ `generate()` allows you to generate data reflecting the null hypothesis, and
+ `calculate()` allows you to calculate a distribution of statistics from the generated data to form the null distribution.

Throughout this vignette, we make use of `gss`, a data set available in infer containing a sample of 500 observations of 11 variables from the *General Social Survey*. 


::: {.cell layout-align="center"}

```{.r .cell-code}
library(tidymodels) # Includes the infer package

# load in the data set
data(gss)

# take a look at its structure
dplyr::glimpse(gss)
#> Rows: 500
#> Columns: 11
#> $ year    <dbl> 2014, 1994, 1998, 1996, 1994, 1996, 1990, 2016, 2000, 1998, 20…
#> $ age     <dbl> 36, 34, 24, 42, 31, 32, 48, 36, 30, 33, 21, 30, 38, 49, 25, 56…
#> $ sex     <fct> male, female, male, male, male, female, female, female, female…
#> $ college <fct> degree, no degree, degree, no degree, degree, no degree, no de…
#> $ partyid <fct> ind, rep, ind, ind, rep, rep, dem, ind, rep, dem, dem, ind, de…
#> $ hompop  <dbl> 3, 4, 1, 4, 2, 4, 2, 1, 5, 2, 4, 3, 4, 4, 2, 2, 3, 2, 1, 2, 5,…
#> $ hours   <dbl> 50, 31, 40, 40, 40, 53, 32, 20, 40, 40, 23, 52, 38, 72, 48, 40…
#> $ income  <ord> $25000 or more, $20000 - 24999, $25000 or more, $25000 or more…
#> $ class   <fct> middle class, working class, working class, working class, mid…
#> $ finrela <fct> below average, below average, below average, above average, ab…
#> $ weight  <dbl> 0.8960034, 1.0825000, 0.5501000, 1.0864000, 1.0825000, 1.08640…
```
:::


Each row is an individual survey response, containing some basic demographic information on the respondent as well as some additional variables. See `?gss` for more information on the variables included and their source. Note that this data (and our examples on it) are for demonstration purposes only, and will not necessarily provide accurate estimates unless weighted properly. For these examples, let's suppose that this data set is a representative sample of a population we want to learn about: American adults.

## Specify variables

The `specify()` function can be used to specify which of the variables in the data set you're interested in. If you're only interested in, say, the `age` of the respondents, you might write:


::: {.cell layout-align="center"}

```{.r .cell-code}
gss %>%
  specify(response = age)
#> Response: age (numeric)
#> # A tibble: 500 × 1
#>      age
#>    <dbl>
#>  1    36
#>  2    34
#>  3    24
#>  4    42
#>  5    31
#>  6    32
#>  7    48
#>  8    36
#>  9    30
#> 10    33
#> # ℹ 490 more rows
```
:::


On the front end, the output of `specify()` just looks like it selects off the columns in the dataframe that you've specified. What do we see if we check the class of this object, though?


::: {.cell layout-align="center"}

```{.r .cell-code}
gss %>%
  specify(response = age) %>%
  class()
#> [1] "infer"      "tbl_df"     "tbl"        "data.frame"
```
:::


We can see that the infer class has been appended on top of the dataframe classes; this new class stores some extra metadata.

If you're interested in two variables (`age` and `partyid`, for example) you can `specify()` their relationship in one of two (equivalent) ways:


::: {.cell layout-align="center"}

```{.r .cell-code}
# as a formula
gss %>%
  specify(age ~ partyid)
#> Response: age (numeric)
#> Explanatory: partyid (factor)
#> # A tibble: 500 × 2
#>      age partyid
#>    <dbl> <fct>  
#>  1    36 ind    
#>  2    34 rep    
#>  3    24 ind    
#>  4    42 ind    
#>  5    31 rep    
#>  6    32 rep    
#>  7    48 dem    
#>  8    36 ind    
#>  9    30 rep    
#> 10    33 dem    
#> # ℹ 490 more rows

# with the named arguments
gss %>%
  specify(response = age, explanatory = partyid)
#> Response: age (numeric)
#> Explanatory: partyid (factor)
#> # A tibble: 500 × 2
#>      age partyid
#>    <dbl> <fct>  
#>  1    36 ind    
#>  2    34 rep    
#>  3    24 ind    
#>  4    42 ind    
#>  5    31 rep    
#>  6    32 rep    
#>  7    48 dem    
#>  8    36 ind    
#>  9    30 rep    
#> 10    33 dem    
#> # ℹ 490 more rows
```
:::


If you're doing inference on one proportion or a difference in proportions, you will need to use the `success` argument to specify which level of your `response` variable is a success. For instance, if you're interested in the proportion of the population with a college degree, you might use the following code:


::: {.cell layout-align="center"}

```{.r .cell-code}
# specifying for inference on proportions
gss %>%
  specify(response = college, success = "degree")
#> Response: college (factor)
#> # A tibble: 500 × 1
#>    college  
#>    <fct>    
#>  1 degree   
#>  2 no degree
#>  3 degree   
#>  4 no degree
#>  5 degree   
#>  6 no degree
#>  7 no degree
#>  8 degree   
#>  9 degree   
#> 10 no degree
#> # ℹ 490 more rows
```
:::


## Declare the hypothesis

The next step in the infer pipeline is often to declare a null hypothesis using `hypothesize()`. The first step is to supply one of "independence" or "point" to the `null` argument. If your null hypothesis assumes independence between two variables, then this is all you need to supply to `hypothesize()`:


::: {.cell layout-align="center"}

```{.r .cell-code}
gss %>%
  specify(college ~ partyid, success = "degree") %>%
  hypothesize(null = "independence")
#> Response: college (factor)
#> Explanatory: partyid (factor)
#> Null Hypothesis: independence
#> # A tibble: 500 × 2
#>    college   partyid
#>    <fct>     <fct>  
#>  1 degree    ind    
#>  2 no degree rep    
#>  3 degree    ind    
#>  4 no degree ind    
#>  5 degree    rep    
#>  6 no degree rep    
#>  7 no degree dem    
#>  8 degree    ind    
#>  9 degree    rep    
#> 10 no degree dem    
#> # ℹ 490 more rows
```
:::


If you're doing inference on a point estimate, you will also need to provide one of `p` (the true proportion of successes, between 0 and 1), `mu` (the true mean), `med` (the true median), or `sigma` (the true standard deviation). For instance, if the null hypothesis is that the mean number of hours worked per week in our population is 40, we would write:


::: {.cell layout-align="center"}

```{.r .cell-code}
gss %>%
  specify(response = hours) %>%
  hypothesize(null = "point", mu = 40)
#> Response: hours (numeric)
#> Null Hypothesis: point
#> # A tibble: 500 × 1
#>    hours
#>    <dbl>
#>  1    50
#>  2    31
#>  3    40
#>  4    40
#>  5    40
#>  6    53
#>  7    32
#>  8    20
#>  9    40
#> 10    40
#> # ℹ 490 more rows
```
:::


Again, from the front-end, the dataframe outputted from `hypothesize()` looks almost exactly the same as it did when it came out of `specify()`, but infer now "knows" your null hypothesis.

## Generate the distribution

Once we've asserted our null hypothesis using `hypothesize()`, we can construct a null distribution based on this hypothesis. We can do this using one of several methods, supplied in the `type` argument:

* `bootstrap`: A bootstrap sample will be drawn for each replicate, where a sample of size equal to the input sample size is drawn (with replacement) from the input sample data.  
* `permute`: For each replicate, each input value will be randomly reassigned (without replacement) to a new output value in the sample.  
* `simulate`: A value will be sampled from a theoretical distribution with parameters specified in `hypothesize()` for each replicate. (This option is currently only applicable for testing point estimates.)  

Continuing on with our example above, about the average number of hours worked a week, we might write:


::: {.cell layout-align="center"}

```{.r .cell-code}
gss %>%
  specify(response = hours) %>%
  hypothesize(null = "point", mu = 40) %>%
  generate(reps = 5000, type = "bootstrap")
#> Response: hours (numeric)
#> Null Hypothesis: point
#> # A tibble: 2,500,000 × 2
#> # Groups:   replicate [5,000]
#>    replicate hours
#>        <int> <dbl>
#>  1         1  53.6
#>  2         1  48.6
#>  3         1  28.6
#>  4         1  46.6
#>  5         1  42.6
#>  6         1  46.6
#>  7         1  58.6
#>  8         1  33.6
#>  9         1  33.6
#> 10         1  23.6
#> # ℹ 2,499,990 more rows
```
:::


In the above example, we take 5000 bootstrap samples to form our null distribution.

To generate a null distribution for the independence of two variables, we could also randomly reshuffle the pairings of explanatory and response variables to break any existing association. For instance, to generate 5000 replicates that can be used to create a null distribution under the assumption that political party affiliation is not affected by age:


::: {.cell layout-align="center"}

```{.r .cell-code}
gss %>%
  specify(partyid ~ age) %>%
  hypothesize(null = "independence") %>%
  generate(reps = 5000, type = "permute")
#> Response: partyid (factor)
#> Explanatory: age (numeric)
#> Null Hypothesis: independence
#> # A tibble: 2,500,000 × 3
#> # Groups:   replicate [5,000]
#>    partyid   age replicate
#>    <fct>   <dbl>     <int>
#>  1 rep        36         1
#>  2 ind        34         1
#>  3 ind        24         1
#>  4 rep        42         1
#>  5 dem        31         1
#>  6 ind        32         1
#>  7 ind        48         1
#>  8 dem        36         1
#>  9 ind        30         1
#> 10 rep        33         1
#> # ℹ 2,499,990 more rows
```
:::


## Calculate statistics

Depending on whether you're carrying out computation-based inference or theory-based inference, you will either supply `calculate()` with the output of `generate()` or `hypothesize()`, respectively. The function, for one, takes in a `stat` argument, which is currently one of `"mean"`, `"median"`, `"sum"`, `"sd"`, `"prop"`, `"count"`, `"diff in means"`, `"diff in medians"`, `"diff in props"`, `"Chisq"`, `"F"`, `"t"`, `"z"`, `"slope"`, or `"correlation"`. For example, continuing our example above to calculate the null distribution of mean hours worked per week:


::: {.cell layout-align="center"}

```{.r .cell-code}
gss %>%
  specify(response = hours) %>%
  hypothesize(null = "point", mu = 40) %>%
  generate(reps = 5000, type = "bootstrap") %>%
  calculate(stat = "mean")
#> Response: hours (numeric)
#> Null Hypothesis: point
#> # A tibble: 5,000 × 2
#>    replicate  stat
#>        <int> <dbl>
#>  1         1  40.0
#>  2         2  40.5
#>  3         3  40.1
#>  4         4  39.9
#>  5         5  39.5
#>  6         6  39.7
#>  7         7  40.3
#>  8         8  39.1
#>  9         9  41.0
#> 10        10  39.1
#> # ℹ 4,990 more rows
```
:::


The output of `calculate()` here shows us the sample statistic (in this case, the mean) for each of our 1000 replicates. If you're carrying out inference on differences in means, medians, or proportions, or $t$ and $z$ statistics, you will need to supply an `order` argument, giving the order in which the explanatory variables should be subtracted. For instance, to find the difference in mean age of those that have a college degree and those that don't, we might write:


::: {.cell layout-align="center"}

```{.r .cell-code}
gss %>%
  specify(age ~ college) %>%
  hypothesize(null = "independence") %>%
  generate(reps = 5000, type = "permute") %>%
  calculate("diff in means", order = c("degree", "no degree"))
#> Response: age (numeric)
#> Explanatory: college (factor)
#> Null Hypothesis: independence
#> # A tibble: 5,000 × 2
#>    replicate    stat
#>        <int>   <dbl>
#>  1         1 -2.21  
#>  2         2 -0.523 
#>  3         3  0.0945
#>  4         4 -0.214 
#>  5         5 -1.01  
#>  6         6  0.0680
#>  7         7 -0.787 
#>  8         8  0.923 
#>  9         9  0.139 
#> 10        10 -0.320 
#> # ℹ 4,990 more rows
```
:::


## Other utilities

The infer package also offers several utilities to extract meaning out of summary statistics and null distributions; the package provides functions to visualize where a statistic is relative to a distribution (with `visualize()`), calculate p-values (with `get_p_value()`), and calculate confidence intervals (with `get_confidence_interval()`).

To illustrate, we'll go back to the example of determining whether the mean number of hours worked per week is 40 hours.


::: {.cell layout-align="center"}

```{.r .cell-code}
# find the point estimate
point_estimate <- gss %>%
  specify(response = hours) %>%
  calculate(stat = "mean")

# generate a null distribution
null_dist <- gss %>%
  specify(response = hours) %>%
  hypothesize(null = "point", mu = 40) %>%
  generate(reps = 5000, type = "bootstrap") %>%
  calculate(stat = "mean")
```
:::


(Notice the warning: `Removed 1244 rows containing missing values.` This would be worth noting if you were actually carrying out this hypothesis test.)

Our point estimate 41.382 seems *pretty* close to 40, but a little bit different. We might wonder if this difference is just due to random chance, or if the mean number of hours worked per week in the population really isn't 40.

We could initially just visualize the null distribution.


::: {.cell layout-align="center"}

```{.r .cell-code}
null_dist %>%
  visualize()
```

::: {.cell-output-display}
![](figs/visualize-1.svg){fig-align='center' width=672}
:::
:::


Where does our sample's observed statistic lie on this distribution? We can use the `obs_stat` argument to specify this.


::: {.cell layout-align="center"}

```{.r .cell-code}
null_dist %>%
  visualize() +
  shade_p_value(obs_stat = point_estimate, direction = "two_sided")
```

::: {.cell-output-display}
![](figs/visualize2-1.svg){fig-align='center' width=672}
:::
:::


Notice that infer has also shaded the regions of the null distribution that are as (or more) extreme than our observed statistic. (Also, note that we now use the `+` operator to apply the `shade_p_value()` function. This is because `visualize()` outputs a plot object from ggplot2 instead of a dataframe, and the `+` operator is needed to add the p-value layer to the plot object.) The red bar looks like it's slightly far out on the right tail of the null distribution, so observing a sample mean of 41.382 hours would be somewhat unlikely if the mean was actually 40 hours. How unlikely, though?


::: {.cell layout-align="center"}

```{.r .cell-code}
# get a two-tailed p-value
p_value <- null_dist %>%
  get_p_value(obs_stat = point_estimate, direction = "two_sided")

p_value
#> # A tibble: 1 × 1
#>   p_value
#>     <dbl>
#> 1    0.04
```
:::


It looks like the p-value is 0.04, which is pretty small---if the true mean number of hours worked per week was actually 40, the probability of our sample mean being this far (1.382 hours) from 40 would be 0.04. This may or may not be statistically significantly different, depending on the significance level $\alpha$ you decided on *before* you ran this analysis. If you had set $\alpha = .05$, then this difference would be statistically significant, but if you had set $\alpha = .01$, then it would not be.

To get a confidence interval around our estimate, we can write:


::: {.cell layout-align="center"}

```{.r .cell-code}
# start with the null distribution
null_dist %>%
  # calculate the confidence interval around the point estimate
  get_confidence_interval(point_estimate = point_estimate,
                          # at the 95% confidence level
                          level = .95,
                          # using the standard error
                          type = "se")
#> # A tibble: 1 × 2
#>   lower_ci upper_ci
#>      <dbl>    <dbl>
#> 1     40.1     42.7
```
:::


As you can see, 40 hours per week is not contained in this interval, which aligns with our previous conclusion that this finding is significant at the confidence level $\alpha = .05$.

## Theoretical methods

The infer package also provides functionality to use theoretical methods for `"Chisq"`, `"F"` and `"t"` test statistics. 

Generally, to find a null distribution using theory-based methods, use the same code that you would use to find the null distribution using randomization-based methods, but skip the `generate()` step. For example, if we wanted to find a null distribution for the relationship between age (`age`) and party identification (`partyid`) using randomization, we could write:


::: {.cell layout-align="center"}

```{.r .cell-code}
null_f_distn <- gss %>%
   specify(age ~ partyid) %>%
   hypothesize(null = "independence") %>%
   generate(reps = 5000, type = "permute") %>%
   calculate(stat = "F")
```
:::


To find the null distribution using theory-based methods, instead, skip the `generate()` step entirely:


::: {.cell layout-align="center"}

```{.r .cell-code}
null_f_distn_theoretical <- gss %>%
   specify(age ~ partyid) %>%
   hypothesize(null = "independence") %>%
   calculate(stat = "F")
```
:::


We'll calculate the observed statistic to make use of in the following visualizations; this procedure is the same, regardless of the methods used to find the null distribution.


::: {.cell layout-align="center"}

```{.r .cell-code}
F_hat <- gss %>% 
  specify(age ~ partyid) %>%
  calculate(stat = "F")
```
:::


Now, instead of just piping the null distribution into `visualize()`, as we would do if we wanted to visualize the randomization-based null distribution, we also need to provide `method = "theoretical"` to `visualize()`.


::: {.cell layout-align="center"}

```{.r .cell-code}
visualize(null_f_distn_theoretical, method = "theoretical") +
  shade_p_value(obs_stat = F_hat, direction = "greater")
```

::: {.cell-output-display}
![](figs/unnamed-chunk-22-1.svg){fig-align='center' width=672}
:::
:::


To get a sense of how the theory-based and randomization-based null distributions relate, we can pipe the randomization-based null distribution into `visualize()` and also specify `method = "both"`


::: {.cell layout-align="center"}

```{.r .cell-code}
visualize(null_f_distn, method = "both") +
  shade_p_value(obs_stat = F_hat, direction = "greater")
```

::: {.cell-output-display}
![](figs/unnamed-chunk-23-1.svg){fig-align='center' width=672}
:::
:::


That's it! This vignette covers most all of the key functionality of infer. See `help(package = "infer")` for a full list of functions and vignettes.


## Session information {#session-info}


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
#>  dplyr      * 1.1.4   2023-11-17 [1] CRAN (R 4.3.1)
#>  ggplot2    * 3.5.0   2024-02-23 [1] CRAN (R 4.3.1)
#>  infer      * 1.0.7   2024-03-25 [1] CRAN (R 4.3.1)
#>  parsnip    * 1.2.1   2024-03-22 [1] CRAN (R 4.3.1)
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
