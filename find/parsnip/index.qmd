---
title: Search parsnip models
weight: 2
description: |
  Find model types, engines, and arguments to fit and predict in the tidymodels framework.
toc: true
toc-depth: 0
listing:
  id: parsnip-list
  contents: items.yml
  template: card.ejs.md
  sort: ["title"]
  sort-ui: [title, model, engine, mode, package]
  filter-ui: true
  categories: false
  page-size: 25
include-after-body: ../../html/resources.html
format:
  html:
    css: ../listing-cards.css
    include-in-header:
      - text: |
          <script src="../listing-filters.js" defer></script>
---

To learn about the parsnip package, see [*Get Started: Build a Model*](/start/models/). Use the filters below to find model types and engines.

<div class="listing-filter-bar"
     data-listing-id="parsnip-list"
     data-fields="model:Model,engine:Engine,mode:Mode,package:Package"></div>

::: {#parsnip-list}
:::
