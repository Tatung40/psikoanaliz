# Genuine native R rsm, one globally shared-step fit of structural rows.
args<-commandArgs(TRUE);stopifnot(length(args)==2)
suppressPackageStartupMessages(library(mirt));stopifnot(as.character(packageVersion('mirt'))=='1.48')
root<-args[1];mode<-args[2];stopifnot(mode %in% c('preflight','fit'));set.seed(20261027)
dat<-read.csv(file.path(root,'stacked.csv'),check.names=FALSE,na.strings='NA');cfg<-jsonlite::read_json(file.path(root,'settings.json'),simplifyVector=TRUE)
grid<-read.csv(file.path(root,'integration.csv'));nodes<-grid$theta;weights<-grid$weight
J<-ncol(dat);C<-cfg$categories;q<-cfg$dimensions;n<-cfg$N
out<-file.path(root,mode);dir.create(out,recursive=TRUE,showWarnings=FALSE)
# Native R gpcmIRT kernel used by itemtype=rsm: increments theta - b_k + c_j.
# Therefore R b_k = Python shared tau_k; R c_j = -Python beta_j (a1=1).
# Disable ONLY default origin/internal ties; explicit model constraints below
# preserve one GLOBAL estimated step vector and Python's first-step-zero gauge.
text<-paste0('F1 = 1-',J)
if(mode=='fit')text<-paste(text,paste0('CONSTRAIN = ',paste(sprintf('(1-%d, b%d)',J,2:(C-1)),collapse=', ')),sep='\n')
spec<-mirt.model(text)
base<-list(data=dat,model=spec,itemtype='rsm',verbose=FALSE,technical=list(internal_constraints=FALSE))
sv<-do.call(mirt,c(base,list(pars='values')));attrs<-attributes(sv)
for(i in seq_len(J)){
 item<-colnames(dat)[i];a<-sv$item==item & sv$name=='a1';loc<-sv$item==item & sv$name=='c'
 stopifnot(sum(a)==1,sum(loc)==1)
 sv$value[a]<-1;sv$est[a]<-FALSE;sv$value[loc]<-0;sv$lbound[loc]<- -6;sv$ubound[loc]<-6;sv$est[loc]<-TRUE
 for(k in seq_len(C-1)){
  index<-sv$item==item & sv$name==paste0('b',k);stopifnot(sum(index)==1)
  sv$value[index]<-k-1;sv$lbound[index]<- -6;sv$ubound[index]<-6;sv$est[index]<-k!=1
 }
}
if(mode=='preflight')sv$est[]<-FALSE
stopifnot(setequal(names(attributes(sv)),names(attrs)),all(vapply(names(attrs),function(k)identical(attr(sv,k),attrs[[k]]),logical(1))))
base$pars<-sv;base$method<-'EM';base$TOL<-1e-8;base$SE<-FALSE;base$accelerate<-'none';base$optimizer<-'L-BFGS-B'
base$technical<-list(internal_constraints=FALSE,customTheta=matrix(nodes,ncol=1),customPriorFun=function(Theta,...)weights,NCYCLES=if(mode=='fit')3000 else 1)
fit<-do.call(mirt,base);values<-mod2values(fit);raw<-coef(fit,simplify=TRUE,IRTpars=FALSE)$items
stopifnot(all(values$value[values$name=='MEAN_1']==0),all(values$value[values$name=='COV_11']==1),!any(values$est[grepl('^(MEAN|COV)_',values$name)]),all(raw[,'a1']==1),all(raw[,'b1']==0))
steps<-raw[,paste0('b',seq_len(C-1)),drop=FALSE]
stopifnot(max(abs(sweep(steps,2,steps[1,],'-')))<8*.Machine$double.eps,max(abs(raw[,'c']))<=6,max(abs(steps))<=6)
actual_prior<-as.numeric(unlist(extract.mirt(fit,'Prior')));stopifnot(length(actual_prior)==61,max(abs(actual_prior-weights))<1e-14,extract.mirt(fit,'logPrior')==0)
write.csv(values,file.path(out,'raw_parameters.csv'),row.names=FALSE)
write.csv(data.frame(b=-raw[,'c']),file.path(out,'parameters.csv'),row.names=FALSE)
write.csv(data.frame(Tau=steps[1,]),file.path(out,'shared_steps.csv'),row.names=FALSE)
theta<-seq(-4,4,length.out=81);curves<-list();integration<-list()
for(i in seq_len(J)){
 item<-extract.item(fit,i);p<-probtrace(item,matrix(theta,ncol=1));info<-iteminfo(item,matrix(theta,ncol=1));gp<-probtrace(item,matrix(nodes,ncol=1))
 curves[[i]]<-data.frame(Node=rep(seq_along(theta)-1,each=C),Item=i-1,Category=rep(0:(C-1),times=length(theta)),Probability=as.vector(t(p)),Information=rep(info,each=C))
 integration[[i]]<-data.frame(Node=rep(seq_along(nodes)-1,each=C),Item=i-1,Category=rep(0:(C-1),times=length(nodes)),Probability=as.vector(t(gp)))
}
write.csv(do.call(rbind,curves),file.path(out,'curves.csv'),row.names=FALSE)
write.csv(do.call(rbind,integration),file.path(out,'integration_probabilities.csv'),row.names=FALSE)
if(mode=='fit'){
 original<-read.csv(file.path(root,'data.csv'),check.names=FALSE);pattern<-as.matrix(read.csv(file.path(root,'pattern.csv')));P<-array(0,c(61,J,C))
 for(i in seq_len(J))P[,i,]<-probtrace(extract.item(fit,i),matrix(nodes,ncol=1))
 scores<-list()
 for(d in seq_len(q))for(r in 1:16){
  ell<-log(weights);for(i in which(pattern[,d]==1))ell<-ell+log(P[,i,original[r,i]+1])
  post<-exp(ell-max(ell));post<-post/sum(post);mu<-sum(post*nodes)
  scores[[length(scores)+1]]<-data.frame(Person=r-1,Factor=d-1,EAP=mu,SD=sqrt(sum(post*(nodes-mu)^2)))
 }
 write.csv(do.call(rbind,scores),file.path(out,'scores.csv'),row.names=FALSE)
}
ll<-extract.mirt(fit,'logLik');aic<-extract.mirt(fit,'AIC');count<-(aic+2*ll)/2
if(mode=='fit')stopifnot(abs(count-(J+C-2))<8*.Machine$double.eps*max(1,abs(aic)))
meta<-list(R=R.version.string,mirt=as.character(packageVersion('mirt')),itemtype='rsm',mode=mode,converged=extract.mirt(fit,'converged'),iterations=extract.mirt(fit,'iterations'),
 logLik=ll,AIC=aic,Native_stacked_BIC=extract.mirt(fit,'BIC'),BIC_original_N=-2*ll+(J+C-2)*log(n),free_parameters=count,
 D=1,mean=0,variance=1,quadrature=61,priors=extract.mirt(fit,'logPrior'),globally_shared_steps=TRUE,first_step_fixed_zero=TRUE,
 bounds=list(b=c(-6,6),tau=c(-6,6)),mapping='R a1=1, b_k=tau_k, c_j=-beta_j; shared steps need not be ordered',
 stopping='EM parameter change <1e-8; L-BFGS-B; max3000; no acceleration',constraints=text)
jsonlite::write_json(meta,file.path(out,'metadata.json'),auto_unbox=TRUE,pretty=TRUE,digits=NA)
capture.output(sessionInfo(),file=file.path(out,'sessionInfo.txt'))
