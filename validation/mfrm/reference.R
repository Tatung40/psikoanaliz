# Rscript reference.R <fixtures dir> <out.csv>   (PROTOCOL.json; mfrmr 0.2.4 MML with estimated N(b0, s2); rasch 1.12.1 PCML)
suppressPackageStartupMessages({ library(mfrmr); library(rasch) })
a <- commandArgs(trailingOnly = TRUE); out <- list()
put <- function(fx, eng, kind, facet, level, key, v) out[[length(out) + 1]] <<- data.frame(fixture = fx, engine = eng,
  kind = kind, facet = facet, level = level, key = key, value = as.numeric(v))
spec <- list(balanced_3facet = "rater_id", unbalanced_linked = "rater_id", four_facet = c("rater_id", "task_id"),
             interaction_bias = "rater_id")
for (fx in names(spec)) {
  d <- read.csv(file.path(a[1], paste0(fx, ".csv")), stringsAsFactors = FALSE)
  fac <- c("item_id", spec[[fx]]); pd <- data.frame(person_id = unique(d$person_id))
  f <- fit_mfrm(d, person = "person_id", facets = fac, score = "score", model = "RSM", method = "MML",
                quad_points = 61, population_formula = ~ 1, person_data = pd, person_id = "person_id")
  put(fx, "mfrmr", "model", "", "", "loglik", f$summary$LogLik); put(fx, "mfrmr", "model", "", "", "converged", f$summary$Converged)
  put(fx, "mfrmr", "model", "", "", "mean", f$population$coefficients[1]); put(fx, "mfrmr", "model", "", "", "variance", f$population$sigma2)
  dg <- diagnose_mfrm(f, diagnostic_mode = "legacy"); m <- as.data.frame(dg$measures)
  for (i in seq_len(nrow(m))) { r <- m[i, ]; fct <- if (r$Facet == "Person") "person" else r$Facet
    for (k in c("Estimate", "ModelSE", "Infit", "Outfit", "ObservedAverage", "ExpectedAverage")) put(fx, "mfrmr", "element", fct, r$Level, k, r[[k]]) }
  st <- as.data.frame(f$steps); for (i in seq_len(nrow(st))) put(fx, "mfrmr", "step", "step", i, "Estimate", st$Estimate[i])
  rl <- as.data.frame(dg$reliability)
  for (i in seq_len(nrow(rl))) for (k in c("SD", "RMSE", "TrueSD", "Separation", "Strata", "Reliability"))
    put(fx, "mfrmr", "reliability", if (rl$Facet[i] == "Person") "person" else rl$Facet[i], "", k, rl[[k]][i])
  if (fx == "interaction_bias") {
    f2 <- fit_mfrm(d, person = "person_id", facets = fac, score = "score", model = "RSM", method = "MML", quad_points = 61,
                   population_formula = ~ 1, person_data = pd, person_id = "person_id", facet_interactions = "item_id:rater_id",
                   min_obs_per_interaction = 1, interaction_policy = "silent")
    put(fx, "mfrmr", "model", "", "", "interaction_loglik", f2$summary$LogLik)
    put(fx, "mfrmr", "model", "", "", "interaction_converged", f2$summary$Converged)
  }
  rr <- rasch_mfrm(d, person = "person_id", item = "item_id", score = "score", facets = spec[[fx]])
  ie <- rr$item_effects; for (i in seq_len(nrow(ie))) put(fx, "rasch", "element", "item_id", ie$item[i], "Estimate", ie$location[i])
  for (fn in names(rr$facet_effects)) { fe <- rr$facet_effects[[fn]]
    for (i in seq_len(nrow(fe))) put(fx, "rasch", "element", fn, fe$level[i], "Estimate", fe$severity[i]) }
}
write.csv(do.call(rbind, out), a[2], row.names = FALSE)
writeLines(c(paste("mfrmr", packageVersion("mfrmr")), paste("rasch", packageVersion("rasch")), R.version.string),
           sub("\\.csv$", "_versions.txt", a[2]))
