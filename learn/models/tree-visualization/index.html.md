---
title: "Tree Visualization"
categories:
  - model fitting
  - pre-processing
  - multivariate analysis
  - correlation
type: learn-subsection
weight: 6
description: | 
  Visualize trees.
  
toc: true
toc-depth: 2
r-packages:
  - tidymodels
  - pls
  - sessioninfo
include-after-body: ../../../html/resources.html
---

## Introduction

To use code in this article,  you will need to install the following packages: pls, sessioninfo, and tidymodels.
Hello

## Session information {#session-info}

::: {.cell layout-align="center"}

```
#> ─ Session info ─────────────────────────────────────────────────────
#>  version  R version 4.5.1 (2025-06-13)
#>  language (EN)
#>  pandoc   3.4
#>  quarto   1.6.42
#> 
#> ─ Packages ─────────────────────────────────────────────────────────
#>  package       version date (UTC)
#>  broom         1.0.13  2026-05-14
#>  dials         1.4.3   2026-04-11
#>  dplyr         1.2.1   2026-04-03
#>  ggplot2       4.0.3   2026-04-22
#>  infer         1.1.0   2025-12-18
#>  parsnip       1.6.0   2026-05-14
#>  pls           2.9-0   2026-02-21
#>  purrr         1.2.2   2026-04-10
#>  recipes       1.3.2   2026-04-02
#>  rlang         1.2.0   2026-04-06
#>  rsample       1.3.2   2026-01-30
#>  sessioninfo   1.2.3   2025-02-05
#>  tibble        3.3.1   2026-01-11
#>  tidymodels    1.5.0   2026-04-23
#>  tune          2.1.0   2026-04-17
#>  workflows     1.3.0   2025-08-27
#>  yardstick     1.4.0   2026-04-07
#> 
#> ────────────────────────────────────────────────────────────────────
```
:::
