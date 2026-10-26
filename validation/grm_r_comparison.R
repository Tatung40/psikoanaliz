# ONE independent R mirt GRM/PCM fit after the actual no-fit preflight.
args <- commandArgs(TRUE)
model <- if(length(args)==4L) args[4] else "GRM"
stopifnot(model %in% c("GRM","PCM","GPCM"),length(args) %in% c(3L,4L),args[3]==paste0("EXECUTE_",model,"_ONCE"),requireNamespace("mirt",quietly=TRUE),requireNamespace("nloptr",quietly=TRUE))
suppressPackageStartupMessages(library(mirt))
stopifnot(as.character(packageVersion("mirt"))=="1.48")
pre <- args[1];out <- args[2];set.seed(20261024)
checks <- read.csv(file.path(pre,"R_checks.csv"))
stopifnot(nrow(checks)>0,all(checks$checked))
data <- read.csv(file.path(pre,"data.csv"),check.names=FALSE)
stopifnot(identical(dim(data),c(500L,8L)),all(vapply(data,function(x) identical(sort(unique(x)),0:3),logical(1))))
type <- if(model=="GRM") "graded" else "gpcm"
saved <- read.csv(file.path(pre,paste0(model,"_configured_layout.csv")),check.names=FALSE)
source("validation/grm_initial_values.R")
sv <- restore_grm_initial_values(data,saved,itemtype=type)
constraints <- read.csv(file.path(pre,paste0(model,"_constraints.csv")),check.names=FALSE)
grid <- read.csv(file.path(pre,"integration.csv"));evaluation <- read.csv(file.path(pre,"evaluation_grid.csv"))
seed <- read.csv(file.path(pre,paste0(model,"_physical.csv")),check.names=FALSE)
free <- which(sv$est);labels <- paste(sv$item[free],sv$name[free],sv$parnum[free],sep=":")
expected_free <- if(model=="PCM") 24L else 32L
stopifnot(identical(as.numeric(sv$parnum),as.numeric(seq_len(nrow(sv)))),length(free)==expected_free,
  identical(names(constraints),c(labels,"rhs")),all(sv$prior.type=="none"),
  all(!sv$est[sv$name %in% c("MEAN_1","COV_11")]),
  sv$value[sv$name=="MEAN_1"]==0,sv$value[sv$name=="COV_11"]==1,
  nrow(grid)==61,all(grid$weight>0),isTRUE(all.equal(sum(grid$weight),1)))
A <- optimizer_constraint_matrix(constraints,labels);rhs <- constraints$rhs
stopifnot(all(as.numeric(A%*%sv$value[free])-rhs<=0))
# Confirm complete common conventional starts in native coordinates.
for(i in seq_len(8L)) {
  b <- as.numeric(unlist(seed[i,c("b1","b2","b3")],use.names=FALSE))
  rows <- which(sv$item==colnames(data)[i])
  stopifnot(identical(seed$item[i],colnames(data)[i]))
  if(model=="GRM") stopifnot(identical(sv$name[rows],c("a1","d1","d2","d3")),
    isTRUE(all.equal(sv$value[rows],c(seed$a[i],-seed$a[i]*b))))
  else stopifnot(identical(sv$name[rows],c("a1",paste0("ak",0:3),paste0("d",0:3))),
    identical(sv$est[rows],c(model=="GPCM",rep(FALSE,5),rep(TRUE,3))),
    isTRUE(all.equal(sv$value[rows],c(seed$a[i],0:3,0,-seed$a[i]*cumsum(b)))))
}
# Redundant intercept boxes: GRM |a*b|<=30; PCM |sum(three steps)|<=18.
limit <- if(model=="PCM") 18 else if(model=="GPCM") 90 else 30
lower <- rep(-limit,length(free));upper <- rep(limit,length(free))
lower[sv$name[free]=="a1"] <- .1;upper[sv$name[free]=="a1"] <- 5
constraint <- function(p,optim_args) {
  stopifnot(identical(unname(which(optim_args$est)),free))
  as.numeric(A%*%p)-rhs
}
jacobian <- function(p,optim_args) {
  stopifnot(identical(unname(which(optim_args$est)),free));A
}
write.csv(sv,file.path(out,"R_starting_values.csv"),row.names=FALSE)
write.csv(data.frame(label=labels,lower=lower,upper=upper),file.path(out,"R_optimizer_boxes.csv"),row.names=FALSE)
if(file.exists(file.path(out,"R_fit_invoked"))) stop("Refit forbidden")
stopifnot(file.create(file.path(out,"R_fit_invoked")))
# This is the ONLY estimation call; no retry, post-hoc initialization or other family.
fit <- mirt(data,1,itemtype=type,pars=sv,method="EM",quadpts=61,
  TOL=1e-8,accelerate="none",SE=FALSE,verbose=FALSE,optimizer="nloptr",
  technical=list(NCYCLES=3000,customTheta=matrix(grid$theta,ncol=1),
    customPriorFun=function(Theta,Etable) grid$weight,storeEMhistory=TRUE,storeEtable=TRUE),
  nloptr_args=list(lb=lower,ub=upper,eval_g_ineq=constraint,eval_jac_g_ineq=jacobian,
    opts=list(algorithm="NLOPT_LD_SLSQP",xtol_rel=1e-8,maxeval=3000,print_level=1)))
capture.output(sessionInfo(),file=file.path(out,"R_sessionInfo.txt"))
source(file.path(out,"pr63_optimizer_trace.R"));write_optimizer_trace(fit,out)
actual_theta <- as.numeric(fit@Internals$Theta);actual_weights <- as.numeric(unlist(extract.mirt(fit,"Prior")))
write.csv(data.frame(theta=actual_theta,weight=actual_weights),file.path(out,"R_actual_integration.csv"),row.names=FALSE)
stopifnot(isTRUE(all.equal(actual_theta,grid$theta)),isTRUE(all.equal(actual_weights,grid$weight)),extract.mirt(fit,"logPrior")==0)
coeff <- coef(fit,simplify=TRUE,IRTpars=FALSE)
stopifnot(as.numeric(coeff$means)==0,as.numeric(coeff$cov)==1)
raw <- coeff$items
if(model!="GRM") {
  stopifnot(all(raw[,"a1"]>0),all(raw[,"d0"]==0))
  if(model=="PCM") stopifnot(all(raw[,"a1"]==1))
  for(c in 0:3) stopifnot(all(raw[,paste0("ak",c)]==c))
  b <- -t(apply(raw[,paste0("d",0:3),drop=FALSE],1,diff))/raw[,"a1"]
} else b <- -raw[,paste0("d",1:3),drop=FALSE]/raw[,"a1"]
parameters <- data.frame(item=colnames(data),a=raw[,"a1"],b1=b[,1],b2=b[,2],b3=b[,3])
write.csv(parameters,file.path(out,"R_parameters.csv"),row.names=FALSE)
write.csv(mod2values(fit),file.path(out,"R_raw_parameters.csv"),row.names=FALSE)
curves <- list()
for(i in seq_len(8L)) {
  item <- extract.item(fit,i)
  P <- probtrace(item,matrix(evaluation$theta,ncol=1));I <- iteminfo(item,matrix(evaluation$theta,ncol=1))
  for(c in 0:3) curves[[length(curves)+1L]] <- data.frame(item=colnames(data)[i],node=evaluation$node,
    theta=evaluation$theta,category=c,probability=P[,c+1],information=as.numeric(I))
}
write.csv(do.call(rbind,curves),file.path(out,"R_curves.csv"),row.names=FALSE)
write.csv(data.frame(engine="R mirt",version=as.character(packageVersion("mirt")),
  R_version=R.version.string,model=model,converged=extract.mirt(fit,"converged"),
  n_parameters=length(free),n_persons=nrow(data),
  iterations=extract.mirt(fit,"iterations"),loglik=extract.mirt(fit,"logLik"),
  aic=extract.mirt(fit,"AIC"),bic=extract.mirt(fit,"BIC"),mean=0,variance=1,
  criterion="all native parameter changes strictly below 1e-8; cap forces false"),
  file.path(out,"R_metadata.csv"),row.names=FALSE)

if(model=="GPCM" && extract.mirt(fit,"converged")) {
  # Independent R EAP definition: the fitted R engine supplies all probabilities.
  # Explicit GH61 weights include the normal prior; no second density multiplier.
  lp <- matrix(log(grid$weight),nrow(data),nrow(grid),byrow=TRUE)
  gh <- list()
  for(i in seq_len(ncol(data))) {
    P <- probtrace(extract.item(fit,i),matrix(grid$theta,ncol=1))
    lp <- lp+t(log(P[,data[[i]]+1L,drop=FALSE]))
    for(c in 0:3) gh[[length(gh)+1L]] <- data.frame(Madde=colnames(data)[i],node=seq_len(nrow(grid))-1L,
      Theta=grid$theta,Kategori=as.character(c),Olasılık=P[,c+1L])
  }
  maxima <- apply(lp,1,max);stopifnot(all(is.finite(maxima)))
  posterior <- exp(sweep(lp,1,maxima,"-"));posterior <- posterior/rowSums(posterior)
  means <- as.numeric(posterior%*%grid$theta)
  deviations <- sweep(matrix(grid$theta,nrow(data),nrow(grid),byrow=TRUE),1,means,"-")
  sds <- sqrt(rowSums(posterior*deviations^2))
  stopifnot(all(is.finite(means)),all(is.finite(sds)),all(sds>0))
  persons <- data.frame(Birey=seq_len(nrow(data)),rowSums(data),means,sds,"EAP",model,check.names=FALSE)
  names(persons) <- c("Birey","Ham puan","Yetenek (θ)","SH (θ)","Yöntem","Model")
  write.csv(persons,file.path(out,"R_persons.csv"),row.names=FALSE)
  write.csv(do.call(rbind,gh),file.path(out,"R_GH_curves.csv"),row.names=FALSE)
}
