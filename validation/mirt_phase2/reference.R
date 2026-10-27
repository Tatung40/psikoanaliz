# Native R mirt reference for one orthogonal between-item block. No new estimator.
args<-commandArgs(TRUE);stopifnot(length(args)==3)
suppressPackageStartupMessages(library(mirt));stopifnot(as.character(packageVersion('mirt'))=='1.48')
model<-args[1];root<-args[2];mode<-args[3];stopifnot(model %in% c('3PL','4PL'),mode %in% c('preflight','fit'))
set.seed(20261027);dat<-read.csv(file.path(root,'data.csv'),check.names=FALSE)
out<-file.path(root,mode);dir.create(out,recursive=TRUE,showWarnings=FALSE)
source('validation/r_mirt_phase1/matched_ml.R')
sv<-mirt(dat,1,itemtype=model,pars='values',verbose=FALSE);native_attributes<-attributes(sv)
opts<-matched_ml(dat,model,sv,root,fixed_asymptotes=TRUE)
stopifnot(setequal(names(attributes(opts$pars)),names(native_attributes)),
 all(vapply(names(native_attributes),function(n)identical(attr(opts$pars,n),native_attributes[[n]]),logical(1))),!any(opts$pars$est[opts$pars$name %in% c('g','u')]))
if(mode=='preflight'){
 opts$pars$est[]<-FALSE;opts$optimizer<-NULL;opts$nloptr_args<-NULL;opts$technical$NCYCLES<-1
}
fit<-do.call(mirt,opts)
final_values<-mod2values(fit);stopifnot(all(final_values$value[final_values$name=='MEAN_1']==0),all(final_values$value[final_values$name=='COV_11']==1),!any(final_values$est[grepl('^(MEAN|COV)_',final_values$name)]))
actual_prior<-as.numeric(unlist(extract.mirt(fit,'Prior')));expected_prior<-read.csv(file.path(root,'integration.csv'))$weight
stopifnot(length(actual_prior)==61,max(abs(actual_prior-expected_prior))<1e-14)
write.csv(mod2values(fit),file.path(out,'raw_parameters.csv'),row.names=FALSE)
theta<-seq(-4,4,length.out=81);nodes<-read.csv(file.path(root,'integration.csv'))$theta
curves<-list();gridp<-list();raw<-coef(fit,simplify=TRUE,IRTpars=FALSE)$items
for(i in seq_len(ncol(dat))){
 item<-extract.item(fit,i);p<-probtrace(item,matrix(theta,ncol=1));I<-iteminfo(item,matrix(theta,ncol=1))
 curves[[i]]<-data.frame(Node=seq_along(theta)-1,Item=i-1,Probability=p[,2],Information=I)
 gridp[[i]]<-probtrace(item,matrix(nodes,ncol=1))[,2]
}
write.csv(do.call(rbind,curves),file.path(out,'curves.csv'),row.names=FALSE)
write.csv(as.data.frame(do.call(cbind,gridp)),file.path(out,'integration_probabilities.csv'),row.names=FALSE)
write.csv(data.frame(a=raw[,'a1'],b=-raw[,'d']/raw[,'a1'],c=raw[,'g'],u=raw[,'u']),file.path(out,'parameters.csv'),row.names=FALSE)
if(mode=='fit'){
 w<-read.csv(file.path(root,'integration.csv'))$weight;P<-do.call(cbind,gridp);scores<-matrix(NA,16,2)
 for(n in 1:16){ell<-log(w);for(i in seq_len(ncol(dat)))ell<-ell+log(if(dat[n,i])P[,i] else 1-P[,i]);post<-exp(ell-max(ell));post<-post/sum(post);mu<-sum(post*nodes);scores[n,]<-c(mu,sqrt(sum(post*(nodes-mu)^2)))}
 write.csv(data.frame(EAP=scores[,1],SD=scores[,2]),file.path(out,'scores.csv'),row.names=FALSE)
}
meta<-list(R=R.version.string,mirt=as.character(packageVersion('mirt')),model=model,mode=mode,
 converged=extract.mirt(fit,'converged'),logLik=extract.mirt(fit,'logLik'),AIC=extract.mirt(fit,'AIC'),BIC=extract.mirt(fit,'BIC'),iterations=extract.mirt(fit,'iterations'),
 fixed_c=.2,fixed_u=if(model=='4PL').95 else 1,mean=final_values$value[final_values$name=='MEAN_1'],variance=final_values$value[final_values$name=='COV_11'],D=1,priors=extract.mirt(fit,'logPrior'),
 bounds=list(a=c(.1,5),b=c(-6,6)),quadrature=61,stopping='R maximum parameter change <1e-8; maximum 3000 EM cycles')
stopifnot(meta$priors==0,all(raw[,'g']==.2),all(raw[,'u']==meta$fixed_u),min(raw[,'a1'])>=.1,max(raw[,'a1'])<=5,max(abs(-raw[,'d']/raw[,'a1']))<=6+1e-7)
jsonlite::write_json(meta,file.path(out,'metadata.json'),auto_unbox=TRUE,pretty=TRUE,digits=NA)
capture.output(sessionInfo(),file=file.path(out,'sessionInfo.txt'))
