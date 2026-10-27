# Optional validation-only R mirt 1.48 configuration; no production defaults.
matched_ml <- function(dat, model, sv, out, initialization=NULL, fixed_asymptotes=FALSE) {
  stopifnot(model %in% c("3PL","4PL"), as.character(packageVersion("mirt"))=="1.48",
            requireNamespace("nloptr",quietly=TRUE))
  source("validation/r_mirt_phase1/physical_b_constraint.R")
  grid <- read.csv(file.path(out,"integration.csv"))
  stopifnot(nrow(grid)==61, all(grid$weight>0), abs(sum(grid$weight)-1)<1e-14)
  sv$value[sv$name=="a1"] <- 1
  sv$value[sv$name=="d"] <- 0
  sv$value[sv$name=="g"] <- .2
  sv$value[sv$name=="u"] <- if(model=="4PL") .95 else 1
  if(!is.null(initialization)) {
    seed <- read.csv(initialization,check.names=FALSE)
    stopifnot(model=="4PL",!anyDuplicated(seed$item),setequal(seed$item,colnames(dat)),
              all(c("a","b","c","d") %in% names(seed)),!anyNA(seed))
    for(i in seq_len(nrow(seed))) {
      p <- seed[i,]
      stopifnot(is.finite(p$a),is.finite(p$b),is.finite(p$c),is.finite(p$d),
        p$a>=.1,p$a<=5,p$b>=-6,p$b<=6,p$c>=0,p$c<=.5,p$d>=.5,p$d<=1)
      # Physical P=c+(u-c)*logistic(a*(theta-b)); R P=g+(u-g)*logistic(a1*theta+d).
      native <- c(a1=p$a,d=-p$a*p$b,g=p$c,u=p$d)
      for(n in names(native)) {
        idx <- sv$item==p$item & sv$name==n
        stopifnot(sum(idx)==1L); sv$value[idx] <- native[[n]]
      }
    }
  }
  if(fixed_asymptotes) sv$est[sv$name %in% c("g","u")] <- FALSE
  sv$value[sv$name=="MEAN_1"] <- 0; sv$value[sv$name=="COV_11"] <- 1
  sv$est[sv$name %in% c("MEAN_1","COV_11")] <- FALSE
  stopifnot(all(sv$est == (sv$name %in% c("a1","d",if(!fixed_asymptotes) "g",if(model=="4PL" && !fixed_asymptotes) "u"))))
  boxes <- list(a1=c(.1,5),d=c(-30,30),g=c(0,.5),u=c(.5,1))
  for(n in names(boxes)) {
    sv$lbound[sv$name==n] <- boxes[[n]][1]
    sv$ubound[sv$name==n] <- boxes[[n]][2]
  }
  stopifnot(all(sv$prior.type=="none"))
  # nloptr receives the compressed, internal logit-coordinate free vector.
  lower <- sv$lbound; upper <- sv$ubound
  idx <- sv$name %in% c("g","u")
  logit <- getFromNamespace("logit","mirt")
  lower[idx] <- logit(lower[idx]); upper[idx] <- logit(upper[idx])
  free <- which(sv$est)
  aidx <- which(sv$name=="a1"); didx <- which(sv$name=="d")
  stopifnot(identical(sv$item[aidx],sv$item[didx]))
  jac <- matrix(0,2*length(aidx),length(free))
  for(i in seq_along(aidx)) {
    jac[c(i,i+length(aidx)),match(aidx[i],free)] <- -6
    jac[i,match(didx[i],free)] <- 1
    jac[i+length(aidx),match(didx[i],free)] <- -1
  }
  callback <- physical_b_constraint(sv)
  internal <- sv$value; internal[idx] <- logit(internal[idx])
  optargs <- list(longpars=internal,est=sv$est)
  p <- internal[free]
  stopifnot(all(callback(p,optargs)<=0))
  # This linear Jacobian check uses exact integer probe coordinates, avoiding
  # subtractive roundoff for imported starts; optimizer starts remain unchanged.
  probe <- p; probe[match(aidx,free)] <- 1; probe[match(didx,free)] <- 0
  baseline <- callback(probe,optargs)
  for(j in seq_along(p)) {
    trial <- probe; trial[j] <- trial[j]+1
    stopifnot(identical(as.numeric(callback(trial,optargs)-baseline),as.numeric(jac[,j])))
  }
  write.csv(sv,file.path(out,"configured_values.csv"),row.names=FALSE)
  list(data=dat,model=1,itemtype=model,pars=sv,method="EM",quadpts=61,
    TOL=1e-8,accelerate="none",SE=FALSE,verbose=FALSE,optimizer="nloptr",
    technical=list(NCYCLES=3000,storeEMhistory=TRUE,customTheta=matrix(grid$theta,ncol=1),
      customPriorFun=function(Theta,Etable) grid$weight),
    nloptr_args=list(lb=lower[free],ub=upper[free],
      eval_g_ineq=callback,
      eval_jac_g_ineq=function(p,optim_args) {
        # mirt's internal est is named; names do not change coordinate order.
        stopifnot(identical(unname(which(optim_args$est)),unname(free))); jac
      },
      opts=list(algorithm="NLOPT_LD_SLSQP",xtol_rel=1e-8,maxeval=3000,print_level=1)))
}
