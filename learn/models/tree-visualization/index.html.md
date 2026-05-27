---
title: "Decision Tree Visualization"
categories:
  - visualization
  - multivariate analysis
  - decision trees
type: learn-subsection
weight: 6
description: | 
  Visualize decision trees using different R packages and approaches.
  
toc: true
toc-depth: 2
r-packages:
  - tidymodels
  - rpart
  - partykit
  - rpart.plot
  - ggplot2
  - parttree
  - rattle
  - sessioninfo
include-after-body: ../../../html/resources.html
---

## Introduction
To use code in this article,  you will need to install the following packages: ggplot2, parttree, partykit, rattle, rpart, rpart.plot, sessioninfo, and tidymodels.

Decision trees are widely used because they are easy to interpret and can
capture nonlinear relationships between predictors and outcomes. Once a
tree model has been fit, visualization becomes an important step for
understanding splits, predictions, and decision boundaries.

R has several different tools for plotting trees. Some produce simple
base R graphics, while others focus on cleaner layouts or integration
with `ggplot2`. This article walks through a few common approaches and
highlights the strengths of each one.

Let's compares and discusses several common methods for visualizing decision
trees in R:

  - [`plot.rpart()`](#base-r-visualization-with-plotrpart)
  - [`partykit::plot()`](#visualization-with-partykit)
  - [`rpart.plot::rpart.plot()`](#visualization-with-rpartplot)
  - [`parttree::geom_parttree()`](#visualization-with-parttree)
  - [`rattle::fancyRpartPlot()`](#visualization-with-fancyrpartplot)
  
We will fit a simple classification tree using the `iris` data set and
compare visualizations results

## Fit a decision tree
We first fit a classification tree using `rpart` through tidymodels.

::: {.cell layout-align="center"}

```{.r .cell-code}
data(iris)

tree_fit <-
  decision_tree(cost_complexity = 0.01) |>
  set_engine("rpart") |>
  set_mode("classification") |>
  fit(Species ~ ., data = iris)

tree_fit
#> parsnip model object
#> 
#> n= 150 
#> 
#> node), split, n, loss, yval, (yprob)
#>       * denotes terminal node
#> 
#> 1) root 150 100 setosa (0.33333333 0.33333333 0.33333333)  
#>   2) Petal.Length< 2.45 50   0 setosa (1.00000000 0.00000000 0.00000000) *
#>   3) Petal.Length>=2.45 100  50 versicolor (0.00000000 0.50000000 0.50000000)  
#>     6) Petal.Width< 1.75 54   5 versicolor (0.00000000 0.90740741 0.09259259) *
#>     7) Petal.Width>=1.75 46   1 virginica (0.00000000 0.02173913 0.97826087) *
```
:::

Several visualization packages work directly with the underlying
`rpart` object, so we extract it here.

::: {.cell layout-align="center"}

```{.r .cell-code}
tree_obj <- extract_fit_engine(tree_fit)
```
:::

## Base R visualization with `plot.rpart()`
The default plotting functions in `rpart` provide a quick way to inspect
the structure of a tree.

::: {.cell layout-align="center"}

```{.r .cell-code}
par(mar = c(2, 2, 2, 2))

plot(tree_obj, uniform = TRUE, margin = 0.1)
text(tree_obj, use.n = TRUE, cex = 0.8)
```

::: {.cell-output-display}
![](figs/plot-rpart-1.svg){fig-align='center' width=100%}
:::
:::

This approach is simple and lightweight, making it useful for quick
exploration. However, the default styling is fairly minimal and can be
hard to customize for presentations or publication-quality graphics.

## Visualization with `partykit`
The `partykit` package provides cleaner default tree visualizations and
more flexible formatting options.

::: {.cell layout-align="center"}

```{.r .cell-code}
party_obj <- as.party(tree_obj)

plot(party_obj)
```

::: {.cell-output-display}
![](figs/plot-partykit-1.svg){fig-align='center' width=672}
:::
:::

Compared to the base `rpart` plot, the layout is often easier to read
and the node labels are displayed more clearly. This makes `partykit`
a good option when readability is important.

`partykit` automatically includes more
information inside the terminal nodes. For example, each terminal node
shows the predicted class along with the distribution of observations
within that node.

The appearance of the plot can also be adjusted with additional plotting
arguments.

::: {.cell layout-align="center"}

```{.r .cell-code}
plot(
  party_obj,
  ip_args = list(abbreviate = FALSE),
  tp_args = list(id = FALSE)
)
```

::: {.cell-output-display}
![](figs/plot-partykit-custom-1.svg){fig-align='center' width=672}
:::
:::

In this example:

- `abbreviate = FALSE` prevents split labels from being shortened
- `id = FALSE` removes terminal node numbers for a cleaner display

These changes are subtle for this example, but they can become
more useful when working with larger trees that contain longer labels or
many terminal nodes.

## Visualization with `rpart.plot`
The `rpart.plot` package is designed specifically for decision tree
visualization and includes several built-in styling improvements.

::: {.cell layout-align="center"}

```{.r .cell-code}
rpart.plot(
  tree_obj,
  roundint = FALSE
)
```

::: {.cell-output-display}
![](figs/plot-rpart-plot-basic-1.svg){fig-align='center' width=672}
:::
:::

The appearance can also be customized with additional arguments.

::: {.cell layout-align="center"}

```{.r .cell-code}
rpart.plot(
  tree_obj,
  type = 4,
  extra = 2,
  under = TRUE,
  faclen = 0,
  roundint = FALSE
)
```

::: {.cell-output-display}
![](figs/plot-rpart-plot-custom-1.svg){fig-align='center' width=672}
:::
:::

This example shows a few ways to adjust the appearance of the tree.

- `type = 4` changes where the split labels are placed. Instead of being
  directly on the branches, the split conditions are displayed above
  them.
- `extra = 2` adds the predicted class label to each terminal node.
- `under = TRUE` places some of the node information underneath the
  node box rather than squeezing everything inside it.
- `faclen = 0` keeps class names from being shortened, so labels like
  `"versicolor"` remain fully visible.
- `roundint = FALSE` suppresses warnings that can appear when plotting
  tidymodels-based trees with `rpart.plot()`.

These options mostly affect how the tree is displayed and labeled.
Different settings may be more useful depending on the complexity of the
tree.

## Visualization with `parttree`
The `parttree` package focuses on visualizing decision boundaries rather
than the tree structure itself. It integrates naturally with `ggplot2`.

::: {.cell layout-align="center"}

```{.r .cell-code}
iris |>
  ggplot(aes(Petal.Length, Petal.Width, color = Species)) +
  geom_point(alpha = 0.7) +
  geom_parttree(data = tree_obj)
```

::: {.cell-output-display}
![](figs/plot-parttree-1.svg){fig-align='center' width=672}
:::
:::

In this plot, the background partitions represent the decision regions
created by the tree model. Each region corresponds to the predicted class
for observations that fall within that section of the predictor space.

Unlike the previous methods, this approach emphasizes how
the model separates observations rather than the exact tree
splits. This can make it easier to understand the overall behavior of
the model, especially for low-dimensional data.

One limitation is that partition plots become harder to interpret as the
number of predictors increases, since only a small number of variables
can be displayed at once.

## Visualization with `fancyRpartPlot()`

The `fancyRpartPlot()` function from the `rattle` package creates a more
stylized version of a decision tree, which looks pretty similar to `rpart.plot`

::: {.cell layout-align="center"}

```{.r .cell-code}
fancyRpartPlot(
  tree_obj,
  sub = ""
)
```

::: {.cell-output-display}
![](figs/plot-fancyrpart-1.svg){fig-align='center' width=672}
:::
:::

The tree is read from top to bottom:

- The top node is called the **root node** and contains all observations
  in the data set.

- Each split asks a question about one predictor variable. For example,
  the first split asks whether `Petal.Length < 2.5`.

- Observations that satisfy the condition follow the left branch labeled
  `"yes"`, while observations that do not satisfy the condition follow
  the right branch labeled `"no"`.

- The nodes at the bottom of the tree are called **terminal nodes** or
  **leaf nodes**. These contain the final predicted class.

In each node:

- The large text at the top shows the predicted species
- The numbers below represent the class probabilities for
  `setosa`, `versicolor`, and `virginica`
- The percentage at the bottom shows the proportion of observations that
  fall into that node

For example, the leftmost terminal node predicts `setosa` with
probability `1.00`, meaning all observations in that node belong to the
`setosa` class.

Compared to the default `rpart` plots, this style is much more visually
distinct and easier to follow at a glance. The use of color also makes
the predicted classes easier to identify. 

One limitation is that the styling can become crowded for larger trees
with many nodes or long labels. However, for smaller and medium-sized
trees, the additional formatting can improve readability.

## Comparing visualization methods

Each visualization approach emphasizes different aspects of the model.

| Method | Strength |
|---|---|
| `plot.rpart()` | Fast and simple, useful for a quick check |
| `partykit` | Cleaner layouts and improved readability |
| `rpart.plot` | Readable and detailed tree summaries |
| `parttree` | Decision boundary visualization for predictor space partitions |
| `fancyRpartPlot()` | More visually distinct and presentation-friendly tree diagrams |

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
#>  parttree      0.1.3   2026-03-31
#>  partykit      1.2-27  2026-03-13
#>  purrr         1.2.2   2026-04-10
#>  rattle        5.6.2   2026-02-08
#>  recipes       1.3.2   2026-04-02
#>  rlang         1.2.0   2026-04-06
#>  rpart         4.1.27  2026-03-27
#>  rpart.plot    3.1.4   2026-01-08
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
