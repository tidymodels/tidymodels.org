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
  # mixOmics must be excluded from the pak call and installed separately via
  # BiocManager — see comment below.
  to_install <- ifelse(needed == "catboost", "catboost/catboost/catboost/R-package", needed)
  to_install <- to_install[to_install != "mixOmics"]
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

  # torch must be installed explicitly for brulee to work
  if ("brulee" %in% needed) {
    cli::cli_alert_info("Installing torch backend for brulee...")
    torch::install_torch()
  }

  # sparklyr requires a local Spark installation
  if ("sparklyr" %in% needed) {
    installed <- sparklyr::spark_installed_versions()
    if (!any(grepl("^3\\.5", installed$spark))) {
      cli::cli_alert_info("Installing Spark for sparklyr...")
      sparklyr::spark_install(version = "3.5")
    }
  }

  cli::cli_alert_success("Done.")
  invisible(failed)
}
