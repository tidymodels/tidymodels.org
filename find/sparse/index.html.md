---
title: Search sparse compatible models and steps
weight: 4
description: | 
 Find recipe steps and model engines that work with sparse data.
toc: true
toc-depth: 0
include-after-body: ../../resources.html
css: ../styles-find.css
---






## Models

Below is all the model engine combinations that support sparse data. It is known that `ranger` supports sparse data as an input, it doesn’t use it any differently than dense data. Thus it shouldn't be used with truly sparse data.



::: {.cell}
::: {.cell-output-display}


```{=html}
<ul>
<li>
<code>boost_tree</code>
</li>
<ul>
<li>
<a href="https://parsnip.tidymodels.org//reference/details_boost_tree_xgboost.html">xgboost</a>
</li>
</ul>
<li>
<code>linear_reg</code>
</li>
<ul>
<li>
<a href="https://parsnip.tidymodels.org//reference/details_linear_reg_glmnet.html">glmnet</a>
</li>
</ul>
<li>
<code>logistic_reg</code>
</li>
<ul>
<li>
<a href="https://parsnip.tidymodels.org//reference/details_logistic_reg_glmnet.html">glmnet</a>
</li>
<li>
<a href="https://parsnip.tidymodels.org//reference/details_logistic_reg_LiblineaR.html">LiblineaR</a>
</li>
</ul>
<li>
<code>multinom_reg</code>
</li>
<ul>
<li>
<a href="https://parsnip.tidymodels.org//reference/details_multinom_reg_glmnet.html">glmnet</a>
</li>
</ul>
<li>
<code>poisson_reg</code>
</li>
<ul>
<li>
<a href="https://parsnip.tidymodels.org//reference/details_poisson_reg_glmnet.html">glmnet</a>
</li>
</ul>
<li>
<code>rand_forest</code>
</li>
<ul>
<li>
<a href="https://parsnip.tidymodels.org//reference/details_rand_forest_ranger.html">ranger</a>
</li>
</ul>
<li>
<code>svm_linear</code>
</li>
<ul>
<li>
<a href="https://parsnip.tidymodels.org//reference/details_svm_linear_LiblineaR.html">LiblineaR</a>
</li>
</ul>
</ul>
```


:::
:::



## Steps

Sparse data compatibility for steps comes in 2 flavors. The first kind generates sparse data from dense data. Often converting categorical variables to many sparse columns. This type of step is listed here:



::: {.cell}
::: {.cell-output-display}


```{=html}
<ul>
<li><a href='https://recipes.tidymodels.org//reference/step_count.html' target='_blank'><tt>step_count</tt></a></li>
<li><a href='https://recipes.tidymodels.org//reference/step_dummy.html' target='_blank'><tt>step_dummy</tt></a></li>
<li><a href='https://recipes.tidymodels.org//reference/step_dummy_extract.html' target='_blank'><tt>step_dummy_extract</tt></a></li>
<li><a href='https://recipes.tidymodels.org//reference/step_dummy_multi_choice.html' target='_blank'><tt>step_dummy_multi_choice</tt></a></li>
<li><a href='https://recipes.tidymodels.org//reference/step_holiday.html' target='_blank'><tt>step_holiday</tt></a></li>
<li><a href='https://recipes.tidymodels.org//reference/step_indicate_na.html' target='_blank'><tt>step_indicate_na</tt></a></li>
<li><a href='https://recipes.tidymodels.org//reference/step_regex.html' target='_blank'><tt>step_regex</tt></a></li>
</ul>
```


:::
:::



The other type of sparse compatible steps are the ones that can take sparse data as input and operate on them while preserving the sparsity. These steps can thus safely be applied to columns that are produced by the above steps. This type of step is listed here:



::: {.cell}
::: {.cell-output-display}


```{=html}
<ul>
<li><a href='https://recipes.tidymodels.org//reference/step_arrange.html' target='_blank'><tt>step_arrange</tt></a></li>
<li><a href='https://recipes.tidymodels.org//reference/step_filter.html' target='_blank'><tt>step_filter</tt></a></li>
<li><a href='https://recipes.tidymodels.org//reference/step_filter_missing.html' target='_blank'><tt>step_filter_missing</tt></a></li>
<li><a href='https://recipes.tidymodels.org//reference/step_impute_mean.html' target='_blank'><tt>step_impute_mean</tt></a></li>
<li><a href='https://recipes.tidymodels.org//reference/step_impute_median.html' target='_blank'><tt>step_impute_median</tt></a></li>
<li><a href='https://recipes.tidymodels.org//reference/step_lag.html' target='_blank'><tt>step_lag</tt></a></li>
<li><a href='https://recipes.tidymodels.org//reference/step_naomit.html' target='_blank'><tt>step_naomit</tt></a></li>
<li><a href='https://recipes.tidymodels.org//reference/step_rename.html' target='_blank'><tt>step_rename</tt></a></li>
<li><a href='https://recipes.tidymodels.org//reference/step_rename_at.html' target='_blank'><tt>step_rename_at</tt></a></li>
<li><a href='https://recipes.tidymodels.org//reference/step_rm.html' target='_blank'><tt>step_rm</tt></a></li>
<li><a href='https://recipes.tidymodels.org//reference/step_sample.html' target='_blank'><tt>step_sample</tt></a></li>
<li><a href='https://recipes.tidymodels.org//reference/step_scale.html' target='_blank'><tt>step_scale</tt></a></li>
<li><a href='https://recipes.tidymodels.org//reference/step_select.html' target='_blank'><tt>step_select</tt></a></li>
<li><a href='https://recipes.tidymodels.org//reference/step_shuffle.html' target='_blank'><tt>step_shuffle</tt></a></li>
<li><a href='https://recipes.tidymodels.org//reference/step_slice.html' target='_blank'><tt>step_slice</tt></a></li>
<li><a href='https://recipes.tidymodels.org//reference/step_sqrt.html' target='_blank'><tt>step_sqrt</tt></a></li>
<li><a href='https://recipes.tidymodels.org//reference/step_zv.html' target='_blank'><tt>step_zv</tt></a></li>
</ul>
```


:::
:::
