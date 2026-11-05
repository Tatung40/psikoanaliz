# Report whether installed TAM / ltm APIs expose an estimated upper asymptote (unrestricted 4PL).
res <- list()
for (p in c("TAM", "ltm")) {
  ok <- requireNamespace(p, quietly=TRUE); res[[p]] <- list(installed=ok)
  if (ok) {
    fns <- ls(getNamespace(p)); res[[p]]$version <- as.character(packageVersion(p))
    cand <- if (p == "TAM") "tam.mml.3pl" else "tpm"
    f <- names(formals(get(cand, envir=getNamespace(p))))
    res[[p]]$function_checked <- cand; res[[p]]$formals <- f
    res[[p]]$upper_asymptote_argument <- any(grepl("slip|upper|est.u|est.d|asymp", f, ignore.case=TRUE))
    res[[p]]$four_pl_functions <- fns[grepl("4pl|fourpl|4PL", fns)]
  }
}
writeLines(jsonlite::toJSON(res, auto_unbox=TRUE, pretty=TRUE), commandArgs(trailingOnly=TRUE)[1])
