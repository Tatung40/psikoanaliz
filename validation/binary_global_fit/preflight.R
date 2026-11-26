# Actual R mirt1.48 no-fit object/parameterization/convention audit.
suppressPackageStartupMessages(library(mirt))
stopifnot(as.character(packageVersion("mirt"))=="1.48")
dir.create("/tmp/global-preflight",showWarnings=FALSE)
dat<-read.csv("tests/fixtures/irt_b11/data_irt3pl_ok.csv")[1:200,]
sv<-mirt(dat,1,itemtype="4PL",pars="values",verbose=FALSE)
for (j in colnames(dat)){
 sv$value[sv$item==j & sv$name=="a1"]<-.8
 sv$value[sv$item==j & sv$name=="d"]<-.16
 sv$value[sv$item==j & sv$name=="g"]<-.15
 sv$value[sv$item==j & sv$name=="u"]<-.85
}
sv$value[sv$name=="MEAN_1"]<-0;sv$value[sv$name=="COV_11"]<-1
sv$est[sv$name %in% c("MEAN_1","COV_11")]<-FALSE
obj<-mirt(dat,1,itemtype="4PL",pars=sv,TOL=NaN,SE=FALSE,verbose=FALSE)
it<-extract.item(obj,1)
theta<-matrix(c(-2,0,2),ncol=1)
actual<-getFromNamespace("ProbTrace","mirt")(it,theta)[,2]
expected<-.15+.7*plogis(.8*theta[,1]+.16)
stopifnot(max(abs(actual-expected))<1e-12)
print(sv[sv$item %in% c(colnames(dat)[1],"GROUP"),])
print(it@par);print(it@parnames);print(obj@OptimInfo)
for(n in c("rmsea","RMSEA.CI","cfi","tli","computeNullModel")){
 cat("R_FUNCTION_BEGIN",n,"\n")
 f<-get0(n,envir=asNamespace("mirt"));if(!is.null(f))print(f) else cat("ABSENT\n")
 cat("R_FUNCTION_END",n,"\n")
}
res<-M2(obj,type="M2",calcNull=FALSE,quadpts=61)
out<-list(R=R.version.string,mirt=as.character(packageVersion("mirt")),pars_class=class(sv),
 physical_probability_error=max(abs(actual-expected)),item_internal=as.list(setNames(it@par,it@parnames)),
 latent=as.list(tail(obj@ParObjects$pars,1)[[1]]@par),M2_fixed_point=as.list(res[1,]))
jsonlite::write_json(out,"/tmp/global-preflight/R.json",auto_unbox=TRUE,pretty=TRUE)
cat("R_PREFLIGHT_JSON",jsonlite::toJSON(out,auto_unbox=TRUE),"\n")
