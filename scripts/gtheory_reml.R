# Original validation/application bridge; no facet/GPL implementation copied.
# Rscript gtheory_reml.R data.csv '<whitelisted design>' output-dir [none|profile]
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==4L, args[4] %in% c('none','profile'))
if (!requireNamespace('lme4', quietly=TRUE) || !requireNamespace('jsonlite', quietly=TRUE))
  stop('R lme4 2.1-0 ve jsonlite gereklidir; otomatik paket kurulmaz.')
if (as.character(utils::packageVersion('lme4')) != '2.1-0') stop('Unvalidated lme4 version')
specs <- list('p x i'=c('p','i'), 'p x r'=c('p','r'),
              'p x i x r'=c('p','i','r','p:i','p:r','i:r'),
              'p x (r:i)'=c('p','i','i:r','p:i'),
              'p x (i:r)'=c('p','r','r:i','p:r'))
residuals <- c('p x i'='p:i','p x r'='p:r','p x i x r'='p:i:r',
               'p x (r:i)'='p:i:r','p x (i:r)'='p:r:i')
design <- args[2]; if (!design %in% names(specs)) stop('Unsupported design')
terms <- specs[[design]]; residual_name <- residuals[[design]]
d <- read.csv(args[1], check.names=FALSE)
for (f in intersect(c('p','i','r'), names(d))) d[[f]] <- factor(d[[f]])
stopifnot(!anyNA(d), all(is.finite(d$score)))
formula <- stats::as.formula(paste('score ~ 1 +', paste(paste0('(1|',terms,')'),collapse=' + ')))
fit <- lme4::lmer(formula, data=d, REML=TRUE,
                 control=lme4::lmerControl(optimizer='bobyqa', optCtrl=list(maxfun=100000)))
component_name <- function(g) {
  if (g=='Residual') return(residual_name)
  for (t in terms) if (setequal(strsplit(g,':',fixed=TRUE)[[1]], strsplit(t,':',fixed=TRUE)[[1]])) return(t)
  stop('Unknown variance group: ',g)
}
vc <- as.data.frame(lme4::VarCorr(fit)); vc <- vc[is.na(vc$var2),]
variance <- data.frame(source=vapply(as.character(vc$grp),component_name,''), Variance=vc$vcov)
variance <- variance[match(c(terms,residual_name),variance$source),]
stopifnot(!anyNA(variance), all(variance$Variance >= 0))
dir.create(args[3],recursive=TRUE,showWarnings=FALSE)
write.csv(variance,file.path(args[3],'variance.csv'),row.names=FALSE)
info <- fit@optinfo; H <- info$derivs$Hessian; grad <- info$derivs$gradient
ev <- if (is.matrix(H)) eigen(H,symmetric=TRUE,only.values=TRUE)$values else NA_real_
hessian_pd <- all(is.finite(ev)) && min(ev)>0
scaled <- if (hessian_pd) as.numeric(solve(chol(H),grad)) else NA_real_
messages <- info$conv$lme4$messages
converged <- (is.null(info$conv$opt) || all(info$conv$opt==0)) && length(messages)==0
diag <- list(converged=converged, singular=lme4::isSingular(fit,tol=1e-4),
             hessian_pd=hessian_pd, hessian_min=if(hessian_pd) min(ev) else NULL,
             max_scaled_gradient=if(all(is.finite(scaled))) max(abs(scaled)) else NULL,
             optimizer='bobyqa', messages=as.list(messages),
             boundary_components=as.list(variance$source[variance$Variance <= .Machine$double.eps]),
             intercept=as.numeric(lme4::fixef(fit)[1]), logLik=as.numeric(logLik(fit)),
             AIC=AIC(fit), BIC=BIC(fit), observations=nrow(d), REML=TRUE,
             versions=list(R=R.version.string,lme4=as.character(packageVersion('lme4')),
                           Matrix=as.character(packageVersion('Matrix')),jsonlite=as.character(packageVersion('jsonlite'))))
jsonlite::write_json(diag,file.path(args[3],'diagnostics.json'),auto_unbox=TRUE,pretty=TRUE,na='null',digits=16)
re <- lme4::ranef(fit,condVar=TRUE)
rr <- do.call(rbind,lapply(names(re),function(g) {
  x <- re[[g]]; variance <- as.numeric(attr(x,'postVar')[1,1,])
  if(any(!is.finite(variance)) || any(variance < 0)) stop('Invalid conditional random-effect variance')
  data.frame(term=component_name(g),level=rownames(x),deviation=x[[1]],conditional_sd=sqrt(variance))
}))
write.csv(rr,file.path(args[3],'random_effects.csv'),row.names=FALSE)
ci <- data.frame(source=character(),lower=numeric(),upper=numeric())
if (args[4]=='profile' && converged && !diag$singular && hessian_pd) {
  intervals <- stats::confint(fit,parm='theta_',method='profile',oldNames=FALSE)
  labels <- rownames(intervals)
  mapped <- vapply(labels,function(label) {
    if (label=='sigma') return(residual_name)
    bits <- strsplit(label,'|',fixed=TRUE)[[1]]
    if(length(bits)!=2L) stop('Unsupported profile CI label: ',label)
    component_name(bits[2])
  },'')
  stopifnot(all(is.finite(intervals)),all(intervals >= 0),!anyDuplicated(mapped),
            setequal(mapped,variance$source),all(intervals[,1] <= intervals[,2]))
  ci <- data.frame(source=mapped,lower=intervals[,1]^2,upper=intervals[,2]^2)
}
write.csv(ci,file.path(args[3],'profile_ci.csv'),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(args[3],'sessionInfo.txt'))
cat('R lme4 REML executed; converged=',converged,' singular=',diag$singular,'\n')
