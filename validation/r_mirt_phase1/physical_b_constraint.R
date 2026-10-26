# Candidate callback for R mirt 1.48 optimizer='nloptr'. NOT a matched-fit profile.
# Source verified: cran/mirt tag 1.48, R/EMstep.utils.R Mstep.optim.
# Univariate, single group, no equality ties; fixed group parameters are omitted
# from p, so reconstruct longpars exactly as Mstep.optim does. Never index p
# directly using mod2values parnum. nloptr requires inequalities <= 0.
physical_b_constraint <- function(values) {
  items <- unique(values$item[values$name == "a1"])
  ai <- vapply(items, function(i) {
    j <- values$parnum[values$item == i & values$name == "a1"]
    stopifnot(length(j) == 1L)
    j
  }, numeric(1))
  di <- vapply(items, function(i) {
    j <- values$parnum[values$item == i & values$name == "d"]
    stopifnot(length(j) == 1L)
    j
  }, numeric(1))
  function(p, optim_args) {
    full <- optim_args$longpars
    full[optim_args$est] <- p
    a <- full[ai]; d <- full[di]
    stopifnot(all(is.finite(a)), all(a > 0), all(is.finite(d)))
    # b=-d/a in [-6,6] iff d-6a <=0 and -d-6a <=0, for a>0.
    c(d - 6*a, -d - 6*a)
  }
}
# Intentionally not wired into reference.R: changing only this constraint would
# leave unmatched asymptote coordinates/endpoints and numerical settings.
