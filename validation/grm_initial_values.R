# Documented mirt initialization: keep the native table; never replay a CSV as pars.
restore_grm_initial_values <- function(data, saved) {
  sv <- mirt::mirt(data,1,itemtype="graded",pars="values",verbose=FALSE)
  stopifnot(identical(names(sv),names(saved)),nrow(sv)==nrow(saved))
  # Everything except existing starts/free flags must remain exactly unchanged.
  for(column in setdiff(names(sv),c("value","est"))) {
    converted <- switch(typeof(sv[[column]]),double=as.double(saved[[column]]),
      integer=as.integer(saved[[column]]),logical=as.logical(saved[[column]]),
      character=as.character(saved[[column]]),stop("Unsupported parameter column type"))
    # R1.48 initializes unused prior hyperparameters as NaN; CSV writes NA.
    # Normalize only a local comparison copy, only when BOTH prior types are none.
    if(column %in% c("prior_1","prior_2")) {
      unused_missing <- sv$prior.type=="none" & saved$prior.type=="none" &
        is.na(sv[[column]]) & is.na(converted)
      converted[unused_missing] <- sv[[column]][unused_missing]
    }
    if(!identical(unname(sv[[column]]),unname(converted)))
      stop(paste("Immutable mirt parameter field mismatch:",column),call.=FALSE)
  }
  stopifnot(!anyNA(saved$value),!anyNA(saved$est))
  sv$value[] <- as.double(saved$value)
  sv$est[] <- as.logical(saved$est)
  sv
}
