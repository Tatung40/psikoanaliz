# Independent R mirt execution only; never reads Python fit results.
# Rscript reference.R MODEL input.csv output_directory
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 3L)
if (!requireNamespace("mirt", quietly=TRUE)) quit(status=42L)
suppressPackageStartupMessages(library(mirt))
set.seed(20261024)
model <- args[1]; dat <- read.csv(args[2], check.names=FALSE)
out <- args[3]; dir.create(out, recursive=TRUE, showWarnings=FALSE)
types <- c("1PL"="2PL", "Rasch"="Rasch", "2PL"="2PL", "3PL"="3PL", "4PL"="4PL",
           "GRM"="graded", "PCM"="gpcm", "GPCM"="gpcm", "NRM"="nominal", "RSM"="rsm")
stopifnot(model %in% names(types), !anyNA(dat))
type <- unname(types[model])
sv <- mirt(dat, 1, itemtype=type, pars="values", verbose=FALSE)
if (model %in% c("1PL", "PCM")) {
  sv$value[sv$name == "a1"] <- 1
  sv$est[sv$name == "a1"] <- FALSE
  # Unlike itemtype='Rasch', these cases retain reference variance exactly 1.
  sv$value[sv$name == "COV_11"] <- 1
  sv$est[sv$name == "COV_11"] <- FALSE
}
if (model %in% c("Rasch", "RSM")) {
  sv$value[sv$name == "a1"] <- 1; sv$est[sv$name == "a1"] <- FALSE
  sv$value[sv$name == "MEAN_1"] <- 0; sv$est[sv$name == "MEAN_1"] <- FALSE
  sv$est[sv$name == "COV_11"] <- TRUE
}
fit <- mirt(dat, 1, itemtype=type, pars=sv, method="EM", quadpts=61,
            TOL=1e-8, technical=list(NCYCLES=3000), SE=TRUE,
            SE.type="Richardson", verbose=FALSE)
theta <- seq(-4, 4, length.out=81)
orientation <- 1
if (model == "NRM") {
  p <- probtrace(extract.item(fit, 1), matrix(c(0,1), ncol=1))
  contrast <- log(p[2,2]/p[2,1]) - log(p[1,2]/p[1,1])
  if (contrast < 0) orientation <- -1
}
curves <- list()
parameters <- list()
emit <- function(i, parameter, values) {
  parameters[[length(parameters)+1L]] <<- data.frame(item=colnames(dat)[i],
    parameter=parameter, component=seq_along(values)-1L, value=as.numeric(values))
}
raw <- coef(fit, simplify=TRUE, IRTpars=FALSE)$items
for (i in seq_len(ncol(dat))) {
  item <- extract.item(fit, i)
  prob <- probtrace(item, matrix(theta * orientation, ncol=1))
  info <- iteminfo(item, matrix(theta * orientation, ncol=1))
  for (c in seq_len(ncol(prob))) curves[[length(curves)+1L]] <- data.frame(
    item=colnames(dat)[i], node=seq_along(theta)-1L, theta=theta, category=c-1L, probability=prob[,c], information=info)
  if (model %in% c("1PL","Rasch","2PL","3PL","4PL")) {
    a <- raw[i,"a1"]
    emit(i,"a",a); emit(i,"b",-raw[i,"d"]/a)
    emit(i,"c",if ("g" %in% colnames(raw)) raw[i,"g"] else 0)
    emit(i,"d",if ("u" %in% colnames(raw)) raw[i,"u"] else 1)
  } else {
    p <- probtrace(item, matrix(c(0,1)*orientation,ncol=1))
    if (model == "GRM") {
      survival <- t(apply(p,1,function(row) rev(cumsum(rev(row)))))[,-1,drop=FALSE]
      logits <- qlogis(survival)
    } else if (model == "NRM") {
      logits <- log(p[,-1,drop=FALSE]/p[,1])
    } else logits <- log(p[,-1,drop=FALSE]/p[,-ncol(p),drop=FALSE])
    slopes <- logits[2,]-logits[1,]; intercepts <- logits[1,]
    if (model == "NRM") {emit(i,"A",slopes); emit(i,"D",intercepts)} else {
      emit(i,"a",slopes); emit(i,"b",-intercepts/slopes)
    }
  }
}
write.csv(do.call(rbind, curves), file.path(out,"curves.csv"), row.names=FALSE)
write.csv(do.call(rbind, parameters),file.path(out,"parameters.csv"),row.names=FALSE)
# Raw values/SEs are retained; NO invalid direct SE transformation/comparison.
write.csv(mod2values(fit), file.path(out,"raw_parameters.csv"), row.names=FALSE)
V <- tryCatch(vcov(fit), error=function(e) NULL)
if (!is.null(V)) write.csv(V, file.path(out,"raw_vcov.csv"))
meta <- data.frame(engine="R mirt", version=as.character(packageVersion("mirt")),
  R_version=R.version.string, model=model, itemtype=type, estimation="ML", D=1,
  converged=extract.mirt(fit,"converged"), loglik=as.numeric(logLik(fit)),
  aic=AIC(fit), bic=BIC(fit), orientation=orientation,
  variance=as.numeric(coef(fit,simplify=TRUE)$cov), points=61, tolerance=1e-8, seed=20261024)
write.csv(meta,file.path(out,"metadata.csv"),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(out,"sessionInfo.txt"))
