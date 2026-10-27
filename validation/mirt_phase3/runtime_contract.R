# NO IRT fits/simulations. Inspect actual pinned package and scalar optimizer API.
args<-commandArgs(TRUE);stopifnot(length(args)==1)
suppressPackageStartupMessages(library(mirt));stopifnot(as.character(packageVersion('mirt'))=='1.48')
out<-args[1];dir.create(out,recursive=TRUE,showWarnings=FALSE)
criterion<-getFromNamespace('hasConverged','mirt');p0<-c(a1=0,g=-999,u=999);p1<-c(a1=1e-9,g=-998,u=998)
stopifnot(criterion(p0,p1,1e-8));p1['a1']<-1e-8;stopifnot(!criterion(p0,p1,1e-8))
optimizer<-get('optim',envir=asNamespace('mirt'));namespace<-environmentName(environment(optimizer))
# EM.group's BL branch chooses nlminb for finite parameter bounds and passes
# that name to its resolved optim(). This probe is a scalar quadratic, NOT IRT.
probe<-tryCatch({x<-optimizer(.1,function(p)sum((p-.2)^2),method='nlminb',lower=-1,upper=1);list(accepted=TRUE,message='Scalar API accepted; no independent BL IRT validation claimed')},error=function(e)list(accepted=FALSE,message=conditionMessage(e)))
result<-list(R=R.version.string,mirt=as.character(packageVersion('mirt')),IRT_fits_executed=FALSE,scalar_quadratic_probe_only=TRUE,
 resolved_optimizer_namespace=namespace,bounded_BL_scalar_probe=probe,
 BL_status=if(probe$accepted)'UNVALIDATED: matched IRT execution still required' else 'NOT COMPARABLE THROUGH THIS BOUNDED BL DISPATCH; no BL fit executed',
 source_correction='reloadPars C++ mutates S4 objects in place; original-pars access alone does NOT prove frozen covariance',
 convergence_rule='Named g/u converted with plogis; every absolute change strictly below1e-8; runtime checks passed')
jsonlite::write_json(result,file.path(out,'report.json'),auto_unbox=TRUE,pretty=TRUE,digits=NA)
capture.output(getFromNamespace('BL.LL','mirt'),file=file.path(out,'BL_objective.txt'))
capture.output(getFromNamespace('EM.group','mirt'),file=file.path(out,'EM_BL_dispatch.txt'))
capture.output(criterion,file=file.path(out,'convergence_rule.txt'))
capture.output(sessionInfo(),file=file.path(out,'sessionInfo.txt'))
cat('RUNTIME_CONTRACT=',jsonlite::toJSON(result,auto_unbox=TRUE),sep='')
