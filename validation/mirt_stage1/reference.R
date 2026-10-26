# Independent R mirt 1.48 reference. ML only; fixed evaluations never re-estimate.
args<-commandArgs(TRUE);stopifnot(length(args)==2)
suppressPackageStartupMessages(library(mirt));stopifnot(as.character(packageVersion('mirt'))=='1.48')
root<-args[1];mode<-args[2];stopifnot(mode %in% c('preflight','fit','evaluate_saved'))
cfg<-jsonlite::fromJSON(file.path(root,'settings.json'));out<-file.path(root,mode)
dir.create(out,recursive=TRUE,showWarnings=FALSE);set.seed(cfg$seed)
dat<-read.csv(file.path(root,'data.csv'),check.names=FALSE);J<-ncol(dat);Q<-cfg$q
binary<-cfg$model %in% c('2PL','Rasch','Bifactor');nominal<-cfg$model=='NRM';unit<-cfg$model %in% c('Rasch','PCM')
graded<-cfg$model=='GRM';partial<-cfg$model %in% c('PCM','GPCM')
correlated<-!identical(cfg$correlated,FALSE);full<-identical(cfg$covariance_mode,'full')
pattern<-as.matrix(cfg$pattern);grid<-read.csv(file.path(root,'integration.csv'));nodes<-as.matrix(grid[,seq_len(Q),drop=FALSE])
trace<-as.matrix(read.csv(file.path(root,'trace.csv')))
syntax<-paste(vapply(seq_len(Q),function(j)paste0('F',j,' = ',paste(which(pattern[,j]!=0),collapse=',')),''),collapse='\n')
spec<-mirt.model(syntax);type<-if(graded)'graded' else if(partial || nominal)'gpcm' else '2PL'
pairs<-do.call(rbind,lapply(2:Q,function(r)cbind(r,seq_len(r-1))))
covariance<-function(eta=0){
 S<-diag(Q)
 if(correlated && full){L<-diag(Q);L[pairs]<-eta;S<-tcrossprod(L);S<-S/outer(sqrt(diag(S)),sqrt(diag(S)))}
 else if(correlated)S[1,2]<-S[2,1]<-tanh(eta)
 S
}
encode_covariance<-function(S){
 if(!full)return(atanh(S[1,2]))
 C<-t(chol(S));L<-C/diag(C);as.numeric(L[pairs])
}
group<-NULL
if(correlated){
 den<-function(obj,Theta){
  stopifnot(identical(dim(Theta),dim(nodes)),max(abs(Theta-nodes))<1e-12)
  S<-covariance(obj@par);C<-tryCatch(chol(S),error=function(e)NULL)
  if(is.null(C) || any(!is.finite(C)))return(rep(1e-300,nrow(Theta)))
  inv<-chol2inv(C);ratio<- -sum(log(diag(C)))-.5*rowSums((Theta%*%inv-Theta)*Theta)
  exp(log(grid$weight)+ratio)
 }
 S<-diag(Q);if(mode=='preflight'){if(full){S[,]<-.25;diag(S)<-1}else S[1,2]<-S[2,1]<-.25}
 eta<-encode_covariance(S);names(eta)<-if(full)paste0('eta',pairs[,1],pairs[,2]) else 'eta12'
 group<-createGroup(par=eta,est=rep(TRUE,length(eta)),den=den,nfact=Q,standardize=FALSE)
}
score_start<-function(i){
 M<-matrix(0,4,Q);M[2,]<-pattern[i,]
 for(c in 2:3)M[c+1,]<-pattern[i,]*.8*sin(c*seq_len(Q))
 M
}
base<-list(data=dat,model=spec,itemtype=type,verbose=FALSE)
if(!is.null(group))base$customGroup<-group
if(nominal)base$gpcm_mats<-lapply(seq_len(J),score_start)
sv<-do.call(mirt,c(base,list(pars='values')))
native_attributes<-attributes(sv)
native_column_classes<-vapply(sv,function(x)paste(class(x),collapse='/'),'')
same_attributes<-function(x,expected){
 observed<-attributes(x)
 identical(sort(names(observed)),sort(names(expected))) &&
 all(vapply(names(expected),function(n)identical(observed[[n]],expected[[n]]),logical(1)))
}
capture.output(dput(native_attributes),file=file.path(out,'parameter_table_attributes.txt'))
stopifnot(is.data.frame(sv),all(sv$prior.type=='none'))
warm<-if(file.exists(file.path(root,'start.csv')))read.csv(file.path(root,'start.csv')) else NULL
A<-matrix(0,0,nrow(sv))
for(i in seq_len(J)){
 ai<-which(sv$item==colnames(dat)[i] & grepl('^a[1-5]$',sv$name));stopifnot(length(ai)==Q)
 a0<-if(cfg$model=='Bifactor')pattern[i,]*c(.7,rep(.5,Q-1)) else pattern[i,]*(if(unit || nominal)1 else 1.2)
 if(!is.null(warm) && !nominal)a0<-as.numeric(unlist(warm[i,paste0('a',seq_len(Q))],use.names=FALSE))
 sv$value[ai]<-a0;sv$est[ai]<-pattern[i,]!=0 & !unit & !nominal
 sv$lbound[ai]<-ifelse(pattern[i,]!=0,if(cfg$model=='Bifactor')-6 else .1,0)
 sv$ubound[ai]<-ifelse(pattern[i,]!=0,if(cfg$model=='Bifactor')6 else 5,0)
 ds<-which(sv$item==colnames(dat)[i] & grepl('^d[0-9]*$',sv$name))
 if(binary){stopifnot(length(ds)==1);sv$value[ds]<-0;sv$lbound[ds]<- -6;sv$ubound[ds]<-6}
 else if(nominal){
  ak<-which(sv$item==colnames(dat)[i] & grepl('^ak',sv$name));stopifnot(length(ak)==4*Q,length(ds)==4)
  M<-score_start(i);mask<-matrix(rep(pattern[i,]!=0,each=4),4,Q);mask[1,]<-FALSE
  sv$value[ak]<-as.vector(M);sv$est[ak]<-as.vector(mask);sv$lbound[ak]<- -5;sv$ubound[ak]<-5
  sv$value[ds]<-0;sv$est[ds]<-c(FALSE,TRUE,TRUE,TRUE);sv$lbound[ds]<- -6;sv$ubound[ds]<-6
 }else{
  s<-sum(a0)
  if(graded){stopifnot(length(ds)==3);sv$value[ds]<- -s*seq(-1,1,length.out=3)}
  else{
   stopifnot(length(ds)==4);ak<-which(sv$item==colnames(dat)[i] & grepl('^ak',sv$name));stopifnot(length(ak)==4)
   sv$value[ak]<-0:3;sv$est[ak]<-FALSE
   sv$value[ds]<-c(0,-s*cumsum(seq(-1,1,length.out=3)));sv$est[ds]<-c(FALSE,TRUE,TRUE,TRUE)
  }
  sv$lbound[ds]<- -30*sum(pattern[i,])*seq_along(ds);sv$ubound[ds]<-30*sum(pattern[i,])*seq_along(ds)
  for(c in seq_len(length(ds)-as.integer(!graded))){
   row<-numeric(nrow(sv));row[ds[if(graded)c else c+1]]<-1
   if(!graded)row[ds[c]]<- -1
   lower<-row;lower[ai]<- -6;upper<- -row;upper[ai]<- -6;A<-rbind(A,lower,upper)
  }
  if(graded)for(c in seq_len(length(ds)-1)){row<-numeric(nrow(sv));row[ai]<-1e-6;row[ds[c+1]]<-1;row[ds[c]]<- -1;A<-rbind(A,row)}
 }
 if(!is.null(warm)){
  if(nominal){rows<-((i-1)*4+1):(i*4);sv$value[ak]<-as.vector(as.matrix(warm[rows,paste0('a',seq_len(Q)),drop=FALSE]));sv$value[ds]<-warm$d[rows]}
  else if(binary)sv$value[ds]<-warm$d[i]
  else {d<-as.numeric(unlist(warm[i,paste0('d',1:3)],use.names=FALSE));sv$value[ds]<-if(graded)d else c(0,cumsum(d))}
 }
}
if(!correlated){
 sv$value[grepl('^MEAN_',sv$name)]<-0;covs<-which(grepl('^COV_',sv$name))
 sv$value[covs]<-vapply(sv$name[covs],function(n){ij<-as.integer(strsplit(sub('COV_','',n),'')[[1]]);as.numeric(ij[1]==ij[2])},0.)
 sv$est[grepl('^(MEAN|COV)_',sv$name)]<-FALSE
}
source_free_parameters<-NA_integer_
if(mode=='evaluate_saved'){
 saved<-read.csv(file.path(root,'fit','raw_parameters.csv'),check.names=FALSE)
 for(n in c('group','item','class','name','parnum'))stopifnot(identical(as.character(saved[[n]]),as.character(sv[[n]])))
 source_free_parameters<-sum(saved$est);sv$value[]<-saved$value
}
if(mode!='fit')sv$est[]<-FALSE
stopifnot(!nrow(A) || max(A%*%sv$value)<=1e-12)
free<-which(sv$est)
callback<-function(p,optim_args){all<-optim_args$longpars;all[optim_args$est]<-p;as.numeric(A%*%all)}
jac<-function(p,optim_args)A[,which(optim_args$est),drop=FALSE]
technical<-list(NCYCLES=if(mode!='fit')1 else cfg$max_iter,customTheta=nodes,storeEMhistory=TRUE,MAXQUAD=20000)
if(!correlated)technical$customPriorFun<-function(Theta,Etable)grid$weight
opts<-c(base,list(pars=sv,method='EM',SE=FALSE,accelerate='none',TOL=cfg$tol,technical=technical))
if(mode=='fit'){
 opts$optimizer<-'nloptr';nl<-list(lb=sv$lbound[free],ub=sv$ubound[free],opts=list(algorithm=if(nominal)'NLOPT_LN_BOBYQA' else 'NLOPT_LD_SLSQP',xtol_rel=1e-8,maxeval=1000))
 if(nrow(A)){nl$eval_g_ineq<-callback;nl$eval_jac_g_ineq<-jac};opts$nloptr_args<-nl
}
capture.output(dput(attributes(sv)),file=file.path(out,'configured_attributes.txt'))
stopifnot(same_attributes(sv,native_attributes),same_attributes(sv,native_attributes[rev(seq_along(native_attributes))]),
 identical(vapply(sv,function(x)paste(class(x),collapse='/'),''),native_column_classes))
bad<-sv;attr(bad,'itemtype')<-'tampered';stopifnot(!same_attributes(bad,native_attributes))
write.csv(sv,file.path(out,'configured_values.csv'),row.names=FALSE)
fit<-do.call(mirt,opts);final<-mod2values(fit);write.csv(final,file.path(out,'raw_parameters.csv'),row.names=FALSE)
get_cov<-function(values){eta<-if(correlated)values$value[grepl('^eta',values$name)] else 0;covariance(eta)}
S<-get_cov(final)
meta<-list(version=as.character(packageVersion('mirt')),R=R.version.string,model=cfg$model,dimensions=Q,
 execution=if(mode=='fit')'CALIBRATION' else if(mode=='evaluate_saved')'SAVED ESTIMATES / NO M-STEP' else 'FIXED PARAMETERS / NO M-STEP',
 source_free_parameters=source_free_parameters,converged=extract.mirt(fit,'converged'),logLik=extract.mirt(fit,'logLik'),
 AIC=extract.mirt(fit,'AIC'),BIC=extract.mirt(fit,'BIC'),iterations=extract.mirt(fit,'iterations'),logPrior=extract.mirt(fit,'logPrior'))
# Canonical sign gauge for signed nominal/bifactor models; keep native estimates.
orientation<-rep(1,Q)
if(nominal || cfg$model=='Bifactor'){
 for(j in seq_len(Q)){
  i<-which(pattern[,j]!=0 & if(nominal)rowSums(pattern)==1 else TRUE)[1]
  if(nominal){ak<-which(final$item==colnames(dat)[i] & grepl('^ak',final$name));value<-matrix(final$value[ak],4,Q)[2,j]}
  else value<-final$value[final$item==colnames(dat)[i] & final$name==paste0('a',j)]
  stopifnot(abs(value)>sqrt(.Machine$double.eps));orientation[j]<-sign(value)
 }
 if(any(orientation!=1)){
  canon<-final
  for(i in seq_len(J)){
   if(nominal){ak<-which(canon$item==colnames(dat)[i] & grepl('^ak',canon$name));M<-matrix(canon$value[ak],4,Q);canon$value[ak]<-as.vector(sweep(M,2,orientation,'*'))}
   else for(j in seq_len(Q)){idx<-which(canon$item==colnames(dat)[i] & canon$name==paste0('a',j));canon$value[idx]<-canon$value[idx]*orientation[j]}
  }
  S<-S*outer(orientation,orientation);if(correlated)canon$value[grepl('^eta',canon$name)]<-encode_covariance(S)
  canon$est[]<-FALSE;fixedopts<-c(base,list(pars=canon,method='EM',SE=FALSE,accelerate='none',TOL=cfg$tol,technical=technical))
  fit<-do.call(mirt,fixedopts);final<-mod2values(fit)
 }
}
meta$covariance<-S;meta$orientation<-orientation
meta$constraints_valid<-(!nrow(A) || max(A%*%final$value)<1e-7) && min(eigen(S,symmetric=TRUE)$values)>0
prior<-as.numeric(unlist(extract.mirt(fit,'Prior')));write.csv(data.frame(weight=prior),file.path(out,'prior.csv'),row.names=FALSE)
curves<-information<-pars<-list()
for(i in seq_len(J)){
 item<-extract.item(fit,i);p<-probtrace(item,trace)
 for(c in seq_len(ncol(p)))curves[[length(curves)+1]]<-data.frame(Node=seq_len(nrow(trace))-1,Item=i-1,Category=c-1,Probability=p[,c])
 if(nominal){
  # Native iteminfo's standard-category formula is not used for custom score
  # matrices. Fisher information comes from actual R probabilities; two steps.
  make_info<-function(h){
   derivatives<-lapply(seq_len(Q),function(j){delta<-rep(0,Q);delta[j]<-h;(probtrace(item,sweep(trace,2,delta,'+'))-probtrace(item,sweep(trace,2,delta,'-')))/(2*h)})
   lapply(seq_len(nrow(trace)),function(n){G<-do.call(cbind,lapply(derivatives,function(d)d[n,]));crossprod(G/sqrt(p[n,]))})
  }
  ii<-make_info(1e-5);check<-make_info(5e-6);stopifnot(max(abs(unlist(ii)-unlist(check)))<1e-8)
 }else ii<-iteminfo(item,trace,multidim_matrix=TRUE)
 for(n in seq_len(nrow(trace)))for(r in seq_len(Q))for(c in seq_len(Q))information[[length(information)+1]]<-data.frame(Node=n-1,Item=i-1,Row=r-1,Column=c-1,Information=ii[[n]][r,c])
 ai<-which(final$item==colnames(dat)[i] & grepl('^a[1-5]$',final$name));ds<-which(final$item==colnames(dat)[i] & grepl('^d[0-9]*$',final$name))
 a<-final$value[ai];d<-final$value[ds]
 if(nominal){
  ak<-which(final$item==colnames(dat)[i] & grepl('^ak',final$name));M<-sweep(matrix(final$value[ak],4,Q),2,a,'*')
  temp<-data.frame(Item=i-1,Category=0:3,d=d);for(j in seq_len(Q))temp[[paste0('a',j)]]<-M[,j];pars[[i]]<-temp
 }else{
  if(partial)d<-diff(d)
  pars[[i]]<-as.data.frame(as.list(c(setNames(a,paste0('a',seq_len(Q))),setNames(d,if(binary)'d' else paste0('d',seq_along(d))))))
 }
}
meta$optimizer<-if(nominal)'R nloptr BOBYQA: native matrix score derivatives are absent in mirt 1.48' else 'R nloptr SLSQP'
meta$information_reference<-if(nominal)'R native gpcm_mats probabilities + checked two-step numerical Fisher' else 'R mirt iteminfo matrix'
write.csv(do.call(rbind,curves),file.path(out,'probabilities.csv'),row.names=FALSE)
write.csv(do.call(rbind,information),file.path(out,'information.csv'),row.names=FALSE)
write.csv(do.call(rbind,pars),file.path(out,'parameters.csv'),row.names=FALSE)
if(mode=='fit'){
 allp<-lapply(seq_len(J),function(i)probtrace(extract.item(fit,i),nodes));rows<-list()
 for(i in seq_len(J))for(c in seq_len(ncol(allp[[i]])))rows[[length(rows)+1]]<-data.frame(Node=seq_len(nrow(nodes))-1,Item=i-1,Category=c-1,Probability=allp[[i]][,c])
 write.csv(do.call(rbind,rows),file.path(out,'integration_probabilities.csv'),row.names=FALSE)
 scoring<-nodes%*%chol(S);p<-lapply(seq_len(J),function(i)probtrace(extract.item(fit,i),scoring));sc<-matrix(NA,16,2*Q)
 for(n in seq_len(16)){
  ell<-log(grid$weight)
  for(i in seq_len(J))ell<-ell+log(pmax(p[[i]][,dat[n,i]+1],1e-300))
  post<-exp(ell-max(ell));post<-post/sum(post);mean<-as.numeric(crossprod(post,scoring))
  sd<-sqrt(colSums((sweep(scoring,2,mean)^2)*post));sc[n,]<-c(mean,sd)
 }
 colnames(sc)<-c(paste0('F',seq_len(Q)),paste0('SD',seq_len(Q)));write.csv(sc,file.path(out,'eap.csv'),row.names=FALSE)
}
jsonlite::write_json(meta,file.path(out,'metadata.json'),auto_unbox=TRUE,pretty=TRUE,digits=NA)
capture.output(sessionInfo(),file=file.path(out,'sessionInfo.txt'))
