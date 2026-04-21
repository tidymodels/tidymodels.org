#!/usr/bin/env Rscript
# Post-render script to normalize volatile H2O/Spark output in rendered .md files.
# Targets:
#   1. H2O model IDs (session-scoped sequential counters)
#   2. Spark model UIDs (random UUIDs per session)
#   3. H2O model summary byte sizes (minor float/serialization noise)

target <- "learn/models/parsnip-predictions/index.html.md"

if (!file.exists(target)) {
  quit(status = 0)
}

lines <- readLines(target, warn = FALSE)

# 1. H2O model IDs: e.g. GBM_model_R_1776455818270_3215
#    Normalize the trailing session counter to 0
lines <- gsub(
  "((?:GBM|GLM|DRF|DeepLearning|NaiveBayes|RuleFit)_model_R_\\d+)_\\d+",
  "\\1_0",
  lines,
  perl = TRUE
)

# 2. Spark UIDs: e.g. gradient_boosted_trees__bfa0ec85_05a5_41d0_aff8_294a1e669c63
#    Normalize the hex UUID to zeros
lines <- gsub(
  "(__)[0-9a-f]{8}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{12}",
  "\\100000000_0000_0000_0000_000000000000",
  lines,
  perl = TRUE
)

# 3. H2O model summary size column: line pattern is
#    #> 1              50                       50               <SIZE>         <DEPTH>
pat3 <- "(#> 1\\s+50\\s+50\\s+)(\\d+)(\\s+\\d+\\s*$)"
m3 <- regexpr(pat3, lines, perl = TRUE)
hits <- which(m3 != -1)
for (i in hits) {
  parts <- regmatches(lines[i], regexec(pat3, lines[i], perl = TRUE))[[1]]
  rounded <- round(as.numeric(parts[3]), -4)
  lines[i] <- paste0(parts[2], rounded, parts[4])
}

writeLines(lines, target)
