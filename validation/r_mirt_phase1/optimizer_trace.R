# Diagnostics only: no fit, optimizer replacement, or stopping-rule modification.
convergence_trace_pair <- function(history, final, iterations) {
  aligned <- nrow(history)==iterations ||
    (nrow(history)==iterations+1L && identical(as.numeric(history[nrow(history),]),as.numeric(final)))
  if(!aligned) return(NULL)
  list(previous=as.numeric(history[iterations,]),current=as.numeric(final))
}

write_optimizer_trace <- function(fit, out) {
  stopifnot(as.character(packageVersion("mirt"))=="1.48")
  history <- mirt::extract.mirt(fit,"EMhistory")
  iterations <- mirt::extract.mirt(fit,"iterations")
  values <- mirt::mod2values(fit)
  values <- values[order(values$parnum),]
  # Version-pinned S4 inspection retains actual internal logit coordinates;
  # inverse-transforming rounded physical 0/1 would lose those coordinates.
  final <- unlist(lapply(fit@ParObjects$pars,function(p) p@par),use.names=FALSE)
  stopifnot(ncol(history)==length(final),nrow(values)==length(final))
  labels <- paste(values$parnum,values$item,values$name,sep=":")
  colnames(history) <- labels
  write.csv(history,file.path(out,"EM_internal_history.csv"),row.names=FALSE)
  write.csv(data.frame(parameter=labels,internal_value=final),
            file.path(out,"final_internal_parameters.csv"),row.names=FALSE)
  capture.output(str(fit@OptimInfo),file=file.path(out,"optimizer_info.txt"))
  criterion <- getFromNamespace("hasConverged","mirt")
  capture.output(criterion,file=file.path(out,"installed_convergence_rule.txt"))
  # EMstep.group.R checks convergence BEFORE appending that cycle's history.
  # An early break thus has iterations rows; a completed cycle has iterations+1.
  pair <- convergence_trace_pair(history,final,iterations)
  if(!is.null(pair)) {
    previous <- pair$previous
    names(previous) <- names(final) <- values$name
    current <- final
    pick <- values$name %in% c("g","u")
    previous[pick] <- plogis(previous[pick]);current[pick] <- plogis(current[pick])
    delta <- abs(current-previous)
    write.csv(data.frame(parameter=labels,previous=unname(previous),current=unname(current),
                         absolute_change=unname(delta),strictly_below_TOL=unname(delta<1e-8)),
              file.path(out,"last_convergence_pair.csv"),row.names=FALSE)
    # Call the installed implementation on the original internal vectors too.
    raw_previous <- pair$previous
    names(raw_previous) <- values$name
    write.csv(data.frame(iterations=iterations,reported_converged=mirt::extract.mirt(fit,"converged"),
      exact_last_pair_satisfied=criterion(raw_previous,final,1e-8),max_change=max(delta),
      at_cycle_cap=iterations==3000),file.path(out,"convergence_trace_summary.csv"),row.names=FALSE)
  } else writeLines("Last pair unavailable: unexpected history/final alignment; do not infer satisfaction.",
                    file.path(out,"convergence_pair_unresolved.txt"))
  ll <- mirt::extract.mirt(fit,"LLhistory")
  write.csv(data.frame(cycle=seq_along(ll),phase="pre-Mstep",loglik=ll),file.path(out,"EM_loglik_history.csv"),row.names=FALSE)
  write.csv(data.frame(loglik=mirt::extract.mirt(fit,"logLik")),
            file.path(out,"final_loglik.csv"),row.names=FALSE)
}
