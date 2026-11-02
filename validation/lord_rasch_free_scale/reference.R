# Actual independent R mirt1.48, free latent variance; exactly four calibrations.
suppressPackageStartupMessages(library(mirt))
stopifnot(as.character(packageVersion('mirt'))=='1.48')
set.seed(20261102)
root <- 'validation/lord_rasch_free_scale/results'
cfg <- jsonlite::read_json(file.path(root,'settings.json'),simplifyVector=TRUE)
wide <- '--wide' %in% commandArgs(trailingOnly=TRUE)
if(wide) {
  # Verify the actual installed implementation before ANY new calibration.
  defaults <- mirt:::makeopts(method='EM',GenRandomPars=FALSE)
  expanded <- mirt:::makeopts(method='EM',GenRandomPars=FALSE,technical=list(theta_lim=c(-12,12)))
  stopifnot(identical(as.numeric(defaults$theta_lim),c(-6,6)),
    identical(as.numeric(expanded$theta_lim),c(-12,12)))
  root<-file.path(root,'wide');dir.create(root,showWarnings=FALSE)
  jsonlite::write_json(list(mirt=as.character(packageVersion('mirt')),default_theta_lim=defaults$theta_lim,
    expanded_theta_lim=expanded$theta_lim,verified_against_actual_runtime=TRUE),
    file.path(root,'runtime_preflight.json'),pretty=TRUE,auto_unbox=TRUE,digits=16)
}
path <- 'tests/fixtures/irt_b10/data_dif1pl_uniform.csv'
stopifnot(digest::digest(file=path,algo='sha256')==cfg$data_sha256)
frame <- read.csv(path,check.names=FALSE)
frame <- frame[complete.cases(frame[,cfg$items,drop=FALSE]),]
stopifnot(nrow(frame)==741L,nrow(frame)==cfg$used_n)
pd <- function(V) {
  V <- (as.matrix(V)+t(as.matrix(V)))/2
  e <- eigen(V,symmetric=TRUE,only.values=TRUE)$values
  stopifnot(all(is.finite(e)),min(e)>0,max(e)/min(e)<=1e8);V
}
emit <- function(x,path)write.csv(x,path,row.names=FALSE)
all_ok <- TRUE
for(q in c(61,101)) {
  out <- file.path(root,paste0('GH',q));dir.create(out,showWarnings=FALSE);vectors<-list();covs<-list()
  for(g in c('R','F')) {
    sub <- frame[frame$Grup==g,cfg$items,drop=FALSE]
    cache <- file.path(out,paste0('r_',g,'.rds'))
    if(file.exists(cache))stop('Fit exists; no automatic repeat')
    sv <- mirt(sub,1,itemtype='Rasch',pars='values',verbose=FALSE)
    stopifnot(inherits(sv,'mirt_df'),all(sv$value[sv$name=='a1']==1),!any(sv$est[sv$name=='a1']))
    sv$value[sv$name=='MEAN_1']<-0;sv$est[sv$name=='MEAN_1']<-FALSE
    sv$value[sv$name=='COV_11']<-1;sv$est[sv$name=='COV_11']<-TRUE
    # Physical bounds: d=-b, variance=SD^2; no slope/intercept coupling needed.
    sv$lbound[sv$name=='d']<- -10;sv$ubound[sv$name=='d']<-10
    sv$lbound[sv$name=='COV_11']<-1e-6;sv$ubound[sv$name=='COV_11']<-400
    tech<-list(NCYCLES=20000)
    if(wide)tech$theta_lim<-c(-12,12)
    fit <- mirt(sub,1,itemtype='Rasch',pars=sv,quadpts=q,TOL=1e-10,
      SE=TRUE,SE.type='Oakes',technical=tech,verbose=FALSE)
    saveRDS(fit,cache)
    values <- mod2values(fit);V <- pd(vcov(fit));labels<-colnames(V)
    emit(values,file.path(out,paste0('r_',g,'_raw_parameters.csv')))
    write.csv(V,file.path(out,paste0('r_',g,'_raw_full_cov.csv')))
    stopifnot(all(values$value[values$name=='a1']==1),!any(values$est[values$name=='a1']),
      values$value[values$name=='MEAN_1']==0,!values$est[values$name=='MEAN_1'],values$est[values$name=='COV_11'])
    ix <- vapply(cfg$items,function(i){j<-which(values$item==i & values$name=='d');stopifnot(length(j)==1L);j},integer(1))
    pos <- match(paste0('d.',values$parnum[ix]),labels);stopifnot(!anyNA(pos))
    b <- -values$value[ix];J<-matrix(0,length(b),ncol(V));J[cbind(seq_along(b),pos)]<- -1
    C <- pd(J%*%V%*%t(J));variance<-values$value[values$name=='COV_11'];sd<-sqrt(variance)
    diag <- list(converged=extract.mirt(fit,'converged'),sd=sd,variance=variance,
      iterations=extract.mirt(fit,'iterations'),loglik=as.numeric(logLik(fit)),
      interior=all(abs(b)<10-1e-6) && sd>1e-3+1e-6 && sd<20-1e-6,
      covariance_valid=TRUE,full_covariance_condition=kappa(V,exact=TRUE),
      identification='a=1; mean=0; SD free',points=q,vcov_labels=labels,
      numerical_limit='Native R regular quadrature differs from native Python GH; both densities checked')
    jsonlite::write_json(diag,file.path(out,paste0('r_',g,'_diagnostics.json')),pretty=TRUE,auto_unbox=TRUE,digits=16)
    emit(data.frame(item=cfg$items,b=b),file.path(out,paste0('r_',g,'_parameters.csv')))
    emit(as.data.frame(C),file.path(out,paste0('r_',g,'_b_cov.csv')))
    vectors[[g]]<-b;covs[[g]]<-C;all_ok<-all_ok && diag$converged && diag$interior
  }
  v<-c(vectors$R,vectors$F);V<-as.matrix(Matrix::bdiag(covs$R,covs$F));k<-length(cfg$items)
  link <- function(v,item) {
    r<-v[seq_len(k)];f<-v[k+seq_len(k)];a<-match(cfg$anchors,cfg$items);i<-match(item,cfg$items)
    (f[i]-mean(f[a]))-(r[i]-mean(r[a]))
  }
  rows<-list();covrows<-list()
  for(item in cfg$studied) {
    delta<-link(v,item);J<-numDeriv::jacobian(function(z)link(z,item),v);C<-pd(J%*%V%*%t(J))
    stat<-as.numeric(delta^2/C[1,1])
    rows[[length(rows)+1]]<-data.frame(Item=item,Statistic=stat,df=1,p=pchisq(stat,1,lower.tail=FALSE),Delta_b=delta,SE_b=sqrt(C[1,1]))
    covrows[[length(covrows)+1]]<-data.frame(Item=item,Row=0,Column=0,Covariance=C[1,1])
  }
  emit(do.call(rbind,rows),file.path(out,'r_lord.csv'));emit(do.call(rbind,covrows),file.path(out,'r_lord_cov.csv'))
}
jsonlite::write_json(list(R=R.version.string,mirt=as.character(packageVersion('mirt')),
  data_sha256=cfg$data_sha256,used_n=nrow(frame),seed=20261102,session=capture.output(sessionInfo())),
  file.path(root,'r_metadata.json'),pretty=TRUE,auto_unbox=TRUE)
stopifnot(all_ok)
