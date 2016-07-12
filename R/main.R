# =============================================
#
#   Main entry point for an end-to-end submission
#
# =============================================
require(mxnet)
require(dplyr)
require(tidyr)

print("Starting main script", quote = FALSE)
print(date())

# -----    Setting model parameters  -----------------------------

model.name <- "modelDoubleConvNN"    # model name

model.eval <- FALSE                 # nb of eval while training
                                    # set to FALSE to disable and 
                                    # train to maximum data

model.lr <- 0.00001                # learning rate

model.round <- 2000                  # nbr of round 

mx.set.seed(42)

print(paste("Using model : ", model.name, ",lr : ",model.lr,", round : ", model.round, ", eval : ", model.eval), quote=FALSE)

# ----------------------------------------------

print("Ensuring data availability", quote=FALSE)
if (!exists("d.train")) {
  if (file.exists("loadData.RData")) {
    print("Loading compressed data")
    load("loadData.RData")
  } else {
    print("Generating data")
    source("loadData.R")
  }
}


#--------------------------------------------------------------------
print(paste0("Asking model ", model.name, " to generate new predictions"), quote=FALSE)

if(exists("preds")) { rm(preds) }
source(paste0("./models/", model.name, ".R"))
if (any(dim(preds) != c(30, 1783))) {
  stop("model file sohould generate a 1783 x 30 data.frame")
}
if (!is.data.frame(preds)) {
  stop("Generated predictions must be a data.frame")
}
colnames(preds) <- 1:1783
rownames(preds) <- (colnames(d.train[, 1:30]))

# ensure predictions are between 0 and 1
preds <-
  apply(
    preds,
    FUN = function(x) {
      min(96, max(1, x))
    },
    MARGIN = c(1, 2)
  )

# --------------------------------------------------------------------
# format and save submission
# --------------------------------------------------------------------
print("Formating submission", quote=FALSE)

p <- tbl_df(preds) %>%
  mutate(FeatureName = colnames(d.train[1, 1:30])) %>%
  gather(key = ImageId, val = prediction,-FeatureName) %>%
  mutate(ImageId = as.integer(ImageId))

s <- d.lookup %>%
  left_join(
    p,
    type = "left",
    match = "first",
    by = c("ImageId", "FeatureName")
  ) %>%
  transmute(RowId = RowId, Location = prediction)

# Saving predictions

print("Saving predictions", quote=FALSE)
sub.file <- paste0("./submissions/sub_", model.name, "-", date(), ".csv")
write.csv(
  x = s,
  file = sub.file,
  quote = FALSE,
  row.names = FALSE
)
print(paste0("Saved file :", sub.file),quote=FALSE)
print(date())
print("Done.", quote=FALSE)
