# Documented mirt initialization: keep the native table; never replay a CSV as pars.
restore_grm_initial_values <- function(data, saved) {
  sv <- mirt::mirt(data,1,itemtype="graded",pars="values",verbose=FALSE)
  stopifnot(identical(names(sv),names(saved)),nrow(sv)==nrow(saved))
  # Everything except existing starts/free flags must remain exactly unchanged.
  for(column in setdiff(names(sv),c("value","est"))) {
    converted <- switch(typeof(sv[[column]]),double=as.double(saved[[column]]),
      integer=as.integer(saved[[column]]),logical=as.logical(saved[[column]]),
      character=as.character(saved[[column]]),stop("Unsupported parameter column type"))
    stopifnot(identical(unname(sv[[column]]),unname(converted)))
  }
  stopifnot(!anyNA(saved$value),!anyNA(saved$est))
  sv$value[] <- as.double(saved$value)
  sv$est[] <- as.logical(saved$est)
  sv
}
