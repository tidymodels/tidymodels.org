# ------------------------------------------------------------------------------
# Install a vector of R packages, handling special cases.
# Source this file and call install_packages(needed).

install_packages <- function(needed) {
  library(cli)

  if (length(needed) == 0) {
    cli::cli_alert_info("No packages to install.")
    return(invisible())
  }

  cli::cli_alert_info("{length(needed)} package{?s} required: {.pkg {needed}}")

  # catboost must be installed from GitHub
  to_install <- ifelse(needed == "catboost", "catboost/catboost/catboost/R-package", needed)
  pak::pak(to_install, upgrade = TRUE)

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
}
