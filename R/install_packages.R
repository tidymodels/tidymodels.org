# ------------------------------------------------------------------------------
# Install a vector of R packages, handling special cases.
# Source this file and call install_packages(needed).

# Install a vector of R packages, handling special cases.
# Returns a character vector of packages that failed to install (invisibly).
install_packages <- function(needed) {
  library(cli)

  failed <- character(0)

  if (length(needed) == 0) {
    cli::cli_alert_info("No packages to install.")
    return(invisible(failed))
  }

  cli::cli_alert_info("{length(needed)} package{?s} required: {.pkg {needed}}")

  # catboost must be installed from GitHub
  # vip was archived from CRAN on 2026-07-08, so install it from GitHub
  # temporarily until it returns to CRAN.
  # mixOmics must be excluded from the pak call and installed separately via
  # install.packages() — see comment below.
  to_install <- ifelse(needed == "catboost", "catboost/catboost/catboost/R-package", needed)
  to_install <- ifelse(to_install == "vip", "bgreenwell/vip", to_install)
  to_install <- to_install[to_install != "mixOmics"]

  # pak's upgrade = TRUE upgrades ALL installed packages, not just those in
  # to_install. So even though "mixOmics" is excluded above, pak will still
  # attempt to upgrade it if a newer version is available — hitting the same
  # subprocess library path issue. Remove it first so pak has nothing to upgrade.
  if ("mixOmics" %in% rownames(utils::installed.packages())) {
    remove.packages("mixOmics")
  }
  pak::pak(to_install, upgrade = TRUE)

  # mixOmics must be installed via install.packages() rather than pak.
  # pak's subprocess environment for source builds uses a temp library path
  # where rlang.so may be compiled against a different R version, causing
  # "undefined symbol: SETLENGTH". BiocManager::install() also triggers this
  # when pak is installed (it uses pak as a backend). Using install.packages()
  # directly with Bioconductor repos avoids the subprocess entirely.
  # Binary packages for new R releases may not be available immediately (e.g.
  # Bioconductor lags a few weeks behind R minor releases), so we catch failures
  # and let callers skip pages that depend on this package rather than aborting
  # the entire render.
  if ("mixOmics" %in% needed) {
    cli::cli_alert_info("Installing mixOmics from Bioconductor via install.packages()...")
    if (!requireNamespace("BiocManager", quietly = TRUE)) {
      install.packages("BiocManager")
    }
    tryCatch(
      install.packages(
        "mixOmics",
        repos = BiocManager::repositories(),
        dependencies = TRUE
      ),
      error = function(e) {
        cli::cli_warn(c(
          "!" = "Failed to install {.pkg mixOmics}: {conditionMessage(e)}",
          "i" = "Pages that require {.pkg mixOmics} will be skipped."
        ))
        failed <<- c(failed, "mixOmics")
      }
    )
  }

  # rstan/rstanarm link against RcppParallel's binary ABI. pak's
  # `upgrade = TRUE` can bump RcppParallel (or StanHeaders/loo) without
  # rebuilding rstan/rstanarm, leaving stale binaries that fail to load
  # ("Please install the rstanarm package to use this engine"). If the
  # Stan stack no longer loads, rebuild it from source so it links against
  # the current dependency versions.
  if ("rstanarm" %in% needed) {
    if (!requireNamespace("rstanarm", quietly = TRUE)) {
      cli::cli_alert_info("rstanarm failed to load; rebuilding rstan/rstanarm from source...")
      install.packages(c("rstan", "rstanarm"), type = "source")
    }
  }

  # torch must be installed explicitly for brulee to work
  if ("brulee" %in% needed) {
    cli::cli_alert_info("Installing torch backend for brulee...")
    torch::install_torch()
  }

  # keras3 requires a Python virtualenv with tensorflow
  if ("keras3" %in% needed) {
    have_venv <- "r-keras" %in% reticulate::virtualenv_list()
    if (!have_venv) {
      cli::cli_alert_info("Installing keras (tensorflow) Python environment...")
      tryCatch(
        keras3::install_keras(backend = "tensorflow"),
        error = function(e) {
          cli::cli_warn(c(
            "!" = "Failed to install keras: {conditionMessage(e)}",
            "i" = "Pages that require {.pkg keras3} will be skipped."
          ))
          failed <<- c(failed, "keras3")
        }
      )
    }
  }

  # sparklyr requires a local Spark installation
  if ("sparklyr" %in% needed) {
    installed <- sparklyr::spark_installed_versions()
    if (!any(grepl("^3\\.5", installed$spark))) {
      cli::cli_alert_info("Installing Spark for sparklyr...")
      tryCatch(
        sparklyr::spark_install(version = "3.5"),
        error = function(e) {
          cli::cli_warn(c(
            "!" = "Failed to install Spark: {conditionMessage(e)}",
            "i" = "Pages that require {.pkg sparklyr} will be skipped."
          ))
          failed <<- c(failed, "sparklyr")
        }
      )
    }
  }

  cli::cli_alert_success("Done.")
  invisible(failed)
}
