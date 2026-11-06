# Rscript gtheoryr_compare.R <gtheoryr source dir> data.csv design outdir — independent reference C (validation only).
# gtheoryr 0.2.0, CRAN mirror cran/gtheoryr @3b53991557bfe2046119d37bf028a4647faedc18 (MIT), sourced: CRAN binary
# is not installable on R 4.3.3. Crossed designs only; its nested model (items within persons) is NOT COMPARABLE.
args <- commandArgs(trailingOnly = TRUE); src <- args[1]; d <- read.csv(args[2]); design <- args[3]; out <- args[4]
for (f in list.files(file.path(src, "R"), full.names = TRUE)) source(f)
gt_require_design <- if (exists("gt_require_design")) gt_require_design else function(...) invisible(TRUE)
dir.create(out, recursive = TRUE, showWarnings = FALSE)
if (design == "p x i") { d <- subset(d, r == 0); g <- gstudy_pxi(d, "p", "i", "score"); D <- function(m) dstudy_pxi(g, n_items = g$n_items * m)
  src_names <- c(person = "p", item = "i", residual = "p x i") }
if (design == "p x r") { d <- subset(d, i == 0); g <- gstudy_pxi(d, "p", "r", "score"); D <- function(m) dstudy_pxi(g, n_items = g$n_items * m)
  src_names <- c(person = "p", item = "r", residual = "p x r") }
if (design == "p x i x r") { g <- gstudy_pxif(d, "p", "i", "r", "score"); D <- function(m) dstudy_pxif(g, n_items = g$n_items * m, n_facets = g$n_facets * m)
  src_names <- c(person = "p", item = "i", facet = "r", "person:item" = "p x i", "person:facet" = "p x r", "item:facet" = "i x r", residual = "p x i x r") }
vc <- g$variance_components; nm <- names(vc); if (is.null(nm)) nm <- names(src_names)
write.csv(data.frame(source = unname(src_names[nm]), value = unname(vc)), file.path(out, "components.csv"), row.names = FALSE)
keep <- setdiff(names(g$ss), "total")
write.csv(data.frame(source = unname(src_names[keep]), SS = unname(g$ss[keep]), df = unname(g$df[keep]), MS = unname(g$ms[keep])),
          file.path(out, "anova.csv"), row.names = FALSE)
rows <- lapply(c(1, 2), function(m) { x <- D(m); data.frame(multiplier = m, relative_error = unname(x$relative_error),
  absolute_error = unname(x$absolute_error), g = unname(x$g_coefficient), phi = unname(x$phi_coefficient)) })
write.csv(do.call(rbind, rows), file.path(out, "gtheoryr.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(out, "sessionInfo.txt"))
