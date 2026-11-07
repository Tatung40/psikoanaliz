# Rscript mfrm_tam.R <long.csv> <outdir> <facets comma list> <nodes> <interaction 0/1>
# Rating-scale many-facet Rasch model by marginal ML: TAM::tam.mml.mfr (Robitzsch, Kiefer & Wu).
# Long data columns: person, item, <facets...>, score (ordered integer categories 0..K).
# Identification: constraint = "items" (item, facet and step effects sum to zero; latent mean and variance free).
suppressPackageStartupMessages(library(TAM))
a <- commandArgs(trailingOnly = TRUE)
long <- read.csv(a[1], stringsAsFactors = FALSE, check.names = FALSE); out <- a[2]
fac <- strsplit(a[3], ",")[[1]]; nodes_n <- as.integer(a[4]); interaction <- a[5] == "1"
dir.create(out, showWarnings = FALSE, recursive = TRUE)
key <- do.call(paste, c(long[c("person", fac)], sep = "\r"))
rows <- unique(data.frame(key = key, long[c("person", fac)], stringsAsFactors = FALSE))
items <- sort(unique(long$item))
resp <- matrix(NA_real_, nrow(rows), length(items), dimnames = list(NULL, items))
resp[cbind(match(key, rows$key), match(long$item, items))] <- long$score
facets <- rows[fac]; for (f in fac) facets[[f]] <- factor(facets[[f]])
form <- as.formula(paste("~ item +", paste(fac, collapse = " + "), "+ step"))
ctrl <- list(nodes = seq(-6, 6, length.out = nodes_n), maxiter = 3000, conv = 1e-7, convD = 1e-7, progress = FALSE,
             acceleration = "Ramsay")
fit <- function(formula) tam.mml.mfr(resp = resp, facets = facets, pid = rows$person, formulaA = formula,
                                       constraint = "items", control = ctrl, verbose = FALSE)
mod <- fit(form)
xs <- mod$xsi.facets; write.csv(data.frame(parameter = xs$parameter, facet = xs$facet, xsi = xs$xsi, se = xs$se.xsi),
                                file.path(out, "parameters.csv"), row.names = FALSE)
w <- tam.wle(mod, progress = FALSE)
write.csv(data.frame(person = mod$person$pid, eap = mod$person$EAP, eap_sd = mod$person$SD.EAP,
                     wle = w$theta, wle_se = w$error, score = mod$person$score, max = mod$person$max),
          file.path(out, "persons.csv"), row.names = FALSE)
s <- c(loglik = -mod$ic$deviance / 2, deviance = mod$ic$deviance, AIC = mod$ic$AIC, BIC = mod$ic$BIC,
       npars = mod$ic$Npars, n_persons = mod$ic$n, mean = as.numeric(mod$beta[1]), variance = as.numeric(mod$variance[1]),
       eap_rel = mod$EAP.rel, wle_rel = w$WLE.rel[1], iterations = mod$iter, maxiter = ctrl$maxiter,
       converged = as.numeric(mod$iter < ctrl$maxiter))
if (interaction && length(fac) >= 1) {
  mod2 <- fit(as.formula(paste("~ item +", paste(fac, collapse = " + "), "+ item:", fac[1], " + step", sep = "")))
  x2 <- mod2$xsi.facets; x2 <- x2[x2$facet == paste0("item:", fac[1]), ]
  write.csv(data.frame(parameter = x2$parameter, xsi = x2$xsi, se = x2$se.xsi), file.path(out, "interaction.csv"), row.names = FALSE)
  s <- c(s, interaction_deviance = mod2$ic$deviance, interaction_npars = mod2$ic$Npars,
         interaction_converged = as.numeric(mod2$iter < ctrl$maxiter))
}
write.csv(data.frame(key = names(s), value = unname(s)), file.path(out, "summary.csv"), row.names = FALSE)
writeLines(c(paste("TAM", as.character(packageVersion("TAM"))), paste("CDM", as.character(packageVersion("CDM"))),
             R.version.string), file.path(out, "versions.txt"))
