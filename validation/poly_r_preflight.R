# R mirt 1.48 fixed-parameter preflight. No optimization or response simulation.
args <- commandArgs(TRUE);stopifnot(length(args)==1L)
out <- args[1]
checks <- list()
check <- function(name, truth) {
  checks[[length(checks)+1L]] <<- data.frame(check=name,checked=isTRUE(truth))
  write.csv(do.call(rbind,checks),file.path(out,"R_checks.csv"),row.names=FALSE)
  if(!isTRUE(truth)) stop(paste("Unresolved preflight mismatch:",name),call.=FALSE)
}
equal <- function(x,y) isTRUE(all.equal(unname(x),unname(y))) # Base R sqrt(eps); algebra only.
stopifnot(requireNamespace("mirt",quietly=TRUE))
suppressPackageStartupMessages(library(mirt))
capture.output(sessionInfo(),file=file.path(out,"R_sessionInfo.txt"))
write.csv(data.frame(R_version=R.version.string,mirt_version=as.character(packageVersion("mirt"))),
          file.path(out,"R_metadata.csv"),row.names=FALSE)
check("R mirt version",as.character(packageVersion("mirt"))=="1.48")
set.seed(20261024)
x <- read.csv(file.path(out,"data.csv"),check.names=FALSE)
C <- 4L;m <- C-1L
check("category codes and per-item support",all(vapply(x,function(col) identical(sort(unique(col)),0:3),logical(1))))
grid <- read.csv(file.path(out,"integration.csv"));theta <- read.csv(file.path(out,"evaluation_grid.csv"))
check("GH61 standard-normal moments",nrow(grid)==61 && all(grid$weight>0) &&
  equal(c(sum(grid$weight),sum(grid$weight*grid$theta),sum(grid$weight*grid$theta^2),sum(grid$weight*grid$theta^4)),c(1,0,1,3)))
technical <- list(customTheta=matrix(grid$theta,ncol=1),customPriorFun=function(Theta,Etable) grid$weight)
opts <- getFromNamespace("makeopts","mirt")(method="EM",quadpts=61,GenRandomPars=FALSE,technical=technical)
check("R accepts explicit GH nodes and prior weights",equal(opts$technical$customTheta,technical$customTheta) &&
      equal(opts$customPriorFun(technical$customTheta,NULL),grid$weight) && opts$quadpts==61)
# This verifies configuration consumption, NOT an EM likelihood execution.
write.csv(data.frame(check="EM integration execution",status="NOT EXECUTED"),file.path(out,"R_GH_scope.csv"),row.names=FALSE)

for(model in c("GRM","PCM")) {
  type <- if(model=="GRM") "graded" else "gpcm"
  # Official no-fit initialization; returns the values table before estimation.
  sv <- mirt(x,1,itemtype=type,pars="values",verbose=FALSE)
  write.csv(sv,file.path(out,paste0(model,"_initial_layout.csv")),row.names=FALSE)
  check(paste(model,"default normal identification"),
    identical(sv$value[sv$name=="MEAN_1"],0) && identical(sv$value[sv$name=="COV_11"],1) &&
    !any(sv$est[sv$name %in% c("MEAN_1","COV_11")]))
  seed <- read.csv(file.path(out,paste0(model,"_physical.csv")),check.names=FALSE)
  check(paste(model,"item names"),identical(seed$item,colnames(x)))
  for(i in seq_len(ncol(x))) {
    rows <- which(sv$item==colnames(x)[i]);names <- sv$name[rows]
    expected <- if(model=="GRM") c("a1",paste0("d",1:m)) else c("a1",paste0("ak",0:m),paste0("d",0:m))
    check(paste(model,i,"native layout"),identical(names,expected))
    b <- as.numeric(unlist(seed[i,paste0("b",1:m)],use.names=FALSE));a <- seed$a[i]
    if(model=="GRM") {
      check(paste(model,i,"ordered thresholds"),a>=.1 && a<=5 && all(b>=-6 & b<=6) && all(diff(b)>=1e-6))
      sv$value[rows] <- c(a,-a*b)
    } else {
      check(paste(model,i,"score anchors"),identical(sv$value[rows[sv$name[rows] %in% paste0("ak",0:m)]],as.numeric(0:m)) &&
        !any(sv$est[rows[sv$name[rows] %in% c(paste0("ak",0:m),"d0")]]))
      sv$value[rows] <- c(1,0:m,0,-cumsum(b));sv$est[rows[1]] <- FALSE
      check(paste(model,i,"PCM discrimination fixed"),sv$value[rows[1]]==1 && !sv$est[rows[1]])
    }
    # Forward/inverse Jacobians: physical thresholds/steps versus native intercepts.
    if(model=="GRM") {
      forward <- function(v) c(v[1],-v[1]*v[-1])
      physical <- c(a,b);J <- rbind(c(1,rep(0,m)),cbind(-b,diag(-a,m)))
      inverse <- rbind(c(1,rep(0,m)),cbind((-a*b)/a^2,diag(-1/a,m)))
    } else {
      forward <- function(v) -cumsum(v)
      physical <- b;J <- -lower.tri(matrix(0,m,m),diag=TRUE)*1
      inverse <- -diag(m);inverse[cbind(2:m,1:(m-1))] <- 1
    }
    numeric_J <- sapply(seq_along(physical),function(j) {
      h <- .Machine$double.eps^(1/3)*max(1,abs(physical[j]))
      up <- dn <- physical;up[j] <- up[j]+h;dn[j] <- dn[j]-h
      (forward(up)-forward(dn))/(2*h)
    })
    check(paste(model,i,"parameter Jacobian"),equal(J,numeric_J) && equal(inverse%*%J,diag(length(physical))))
    write.csv(J,file.path(out,paste0(model,"_",i,"_forward_J.csv")),row.names=FALSE)
  }
  check(paste(model,"no item priors"),all(sv$prior.type=="none"))
  write.csv(sv,file.path(out,paste0(model,"_configured_layout.csv")),row.names=FALSE)
  if(model=="GRM") {
    source("validation/grm_initial_values.R")
    source("tests/test_grm_initial_values.R")
    check("GRM native initialization survives CSV replay",test_grm_initial_values(x,sv,out))
  }
  free <- which(sv$est);A <- matrix(numeric(),nrow=0,ncol=length(free));rhs <- numeric()
  add <- function(rows,coeff,value=0) {
    row <- numeric(length(free));idx <- match(rows,free)
    check(paste(model,"constraint free-coordinate lookup"),!anyNA(idx))
    row[idx] <- coeff;A <<- rbind(A,row);rhs <<- c(rhs,value)
  }
  for(item in colnames(x)) {
    ix <- which(sv$item==item);ix <- setNames(ix,sv$name[ix]);ds <- ix[paste0("d",1:m)]
    if(model=="GRM") {
      ai <- ix[["a1"]];add(ai,1,5);add(ai,-1,-.1)
      for(d in ds) {add(c(d,ai),c(1,-6));add(c(d,ai),c(-1,-6))}
      for(s in 1:(m-1)) add(c(ds[s+1],ds[s],ai),c(1,-1,1e-6))
    } else for(s in 1:m) {
      rows <- if(s==1) ds[s] else c(ds[s],ds[s-1]);coeff <- if(s==1) 1 else c(1,-1)
      add(rows,coeff,6);add(rows,-coeff,6)
    }
  }
  p <- sv$value[free]
  check(paste(model,"constraint feasibility"),all(as.numeric(A%*%p)-rhs<=0))
  numeric_A <- sapply(seq_along(p),function(j) {q<-p;q[j]<-q[j]+1;as.numeric(A%*%q-A%*%p)})
  check(paste(model,"constraint Jacobian"),equal(A,numeric_A))
  colnames(A) <- paste(sv$item[free],sv$name[free],sv$parnum[free],sep=":")
  write.csv(cbind(as.data.frame(A),rhs=rhs),file.path(out,paste0(model,"_constraints.csv")),row.names=FALSE)
  # Feasible and deliberately infeasible witnesses, without clipping or sorting.
  witness <- function(a,b) {
    q<-sv$value;ix<-which(sv$item==colnames(x)[1]);ix<-setNames(ix,sv$name[ix])
    if(model=="GRM") {q[ix['a1']]<-a;q[ix[paste0('d',1:m)]]<--a*b}
    else q[ix[paste0('d',1:m)]]<--cumsum(b)
    residual <- as.numeric(A%*%q[free])-rhs
    if(model=="GRM" && a %in% c(.1,5)) {
      cat("GRM boundary witness diagnostics: ",file=stderr())
      dput(list(a=a,b=b,native=q[ix[paste0('d',1:m)]],
        positive=which(residual>0),residual=residual[residual>0]),file=stderr())
    }
    all(residual<=0)
  }
  check(paste(model,"physical box witnesses"),witness(1,c(-6,0,6)) && !witness(1,c(-6,0,6.1)))
  if(model=="GRM") check("GRM ordering/slope witnesses",!witness(1,c(0,0,1)) && !witness(5.1,c(-1,0,1)) &&
    witness(.1,c(-6,0,6)) && witness(5,c(-6,0,6)))
  else check("PCM steps need not be ordered",witness(1,c(.5,-.5,.2)))
  curves <- list()
  for(i in seq_len(ncol(x))) {
    rows <- which(sv$item==colnames(x)[i])
    # Same S4 constructor pattern used by R mirt1.48's own item machinery.
    # No simdata(), fit, response draw, or zero-iteration model fit is called.
    obj <- methods::new(type,par=as.numeric(sv$value[rows]),nfact=1L,ncat=C)
    if(type=="gpcm") obj@mat <- FALSE
    P <- probtrace(obj,matrix(theta$theta,ncol=1));I <- iteminfo(obj,matrix(theta$theta,ncol=1))
    b <- as.numeric(unlist(seed[i,paste0('b',1:m)],use.names=FALSE));a <- seed$a[i]
    if(model=="GRM") {
      S <- cbind(1,plogis(outer(theta$theta,b,'-')*a),0)
      expected <- S[,1:C]-S[,2:(C+1)]
      derivative <- a*S*(1-S);dP<-derivative[,1:C]-derivative[,2:(C+1)]
      info<-rowSums(dP^2/expected)
    } else {
      logits <- outer(theta$theta,0:m)+matrix(c(0,-cumsum(b)),nrow(theta),C,byrow=TRUE)
      E <- exp(logits-apply(logits,1,max));expected <- E/rowSums(E)
      mean <- as.numeric(expected%*%(0:m));info<-rowSums(expected*(matrix(0:m,nrow(theta),C,byrow=TRUE)-mean)^2)
    }
    check(paste(model,i,"R category probabilities"),equal(P,expected) && all(is.finite(P)) && all(P>=0) && equal(rowSums(P),rep(1,nrow(P))))
    check(paste(model,i,"R item information"),equal(as.numeric(I),info) && all(is.finite(I)))
    for(c in 0:m) curves[[length(curves)+1L]] <- data.frame(item=colnames(x)[i],node=theta$node,theta=theta$theta,
      category=c,probability=P[,c+1],information=as.numeric(I))
  }
  write.csv(do.call(rbind,curves),file.path(out,paste0(model,"_R_curves.csv")),row.names=FALSE)
}
writeLines("PREFLIGHT CHECKED ONLY. No model fits; scientific PASS unavailable.",file.path(out,"R_scope.txt"))
