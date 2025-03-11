<a alt = "Netlify Deployments" href="https://app.netlify.com/sites/tidymodels-org/deploys"><img src="https://api.netlify.com/api/v1/badges/1979930f-1fd5-42cd-a097-c582d16c24d9/deploy-status" height = 20 /></a>
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" height = 20 /></a>

# tidymodels.org

This repo is the source of <https://www.tidymodels.org>, and this readme tells you how it all works. 

* If you spot any small problems with the website, please feel empowered to fix 
  them directly with a PR. 
  
* If you see any larger problems, an issue is probably better: that way we can 
  discuss the problem before you commit any time to it.

This repo (and resulting website) is licensed as [CC BY-SA](LICENSE.md).

## Requirements to preview the site locally 

### R packages

When updating the site, the goal is to use the most recent CRAN versions of the modeling/data analysis packages. 

1. Get a local copy of the website source.
   * Users of devtools/usethis can do:
     ```r
     usethis::create_from_github("tidymodels/tidymodels.org")
     ```
     Note that `usethis::create_from_github()` works best when it can find a
     GitHub personal access token and usethis (git2r, really) is configured
     correctly for your preferred transport protocol (SSH vs HTTPS).
     [Setup advice](https://usethis.r-lib.org/articles/articles/usethis-setup.html).
   * Otherwise, use your favorite method to fork and clone or download the
     repo as a ZIP file and unpack.
   
1. Start R in your new `tidymodels.org/` directory. 
   
1. To install the required packages, run the code within
   
   ```
   installs.R
   ```
   
   This file will also install the `keras` python libraries and environments. 
   
1. Restart R.

1. You should now be able to render the site in all the usual ways for quarto by calling `quarto render`.

### Quarto

We use the latest release version of quarto. You can install and manage different version with [qvm](https://github.com/dpastoor/qvm).

The website is set up to render with [Netlify](https://app.netlify.com/sites/tidymodels-org/deploys) in according to [quarto documentation](https://quarto.org/docs/publishing/netlify.html).

The files [`_publish.yml`](_publish.yml), [`netlify.toml`](netlify.toml), and [`package.json`](package.json) specifies this configuration. 

## Structure

The source of the website is a collection of `.qmd` files stored in the folders in this repository. This site is then rendered as a [Quarto html website](https://quarto.org/docs/websites/). 

* [`packages/`](packages/): this is a top-level page on the site rendered from a single `.qmd` file.
  
* [`start/`](start/): these files make up a 5-part tutorial series to help users get started with tidymodels. Each article is an `.qmd` file as a page bundle, meaning that each article is in its own folder along with accompanying images, data, and rendered figures.
  
* [`learn/`](learn/): these files make up the articles presented in the learn section. This section is nested, meaning that inside this section, there are actually 4 subsections: `models`, `statistics`, `work`, `develop`. Each article is an `.qmd` file.

* [`help/`](help/): this is a top-level page on the site rendered from a single `.qmd` file.

* [`contribute/`](contribute/): this is a top-level page on the site rendered from a single `.qmd` file.

* [`books/`](books/): these files make up the books page, linked from resource stickies. To add a new book, create a new folder with a new `.qmd` file inside named `index.qmd`. An image file of the cover should be added in the same folder, named `cover.*`.

* [`find/`](find/): these files make up the find page, linked from the top navbar and resource stickies. Each of these pages is an `.qmd` file.


## Workflow

* To add a new post to `learn/`, add a new folder with a `index.qmd` file in it and adapt the YAML header from an existing post. If new packages are required to run this post, then add them to the `packages` object in `installs.R`.

* To preview the site, render it locally with the latest quarto release version.

* The site is published via Netlify but rendered locally, so add those files to the PR. 

* To do a complete rerender, run `re-render.R` script.

## Rerender

We try to do a rerender after a release of a main package.

* Make sure that `all_packages.R` is up to date.

* Run `installs.R` script. Make sure to check that dev versions aren't present.

* Run `re-render.R` script.