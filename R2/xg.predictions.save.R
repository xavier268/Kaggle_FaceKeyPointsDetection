
# ==============================
# Save predicions to file
# ==============================

# input : prediction is an object where key points are organized by rows, in the same format as d.train

xg.predictions.save <- function(predictions) {
  d.lookup <- colnames(d.test)
  submission <- read.csv(file = "../Dataset/IdLookupTable.csv")
  
  for(i in 1:nrow(submission)){
    imId <- submission$ImageId[i]
    feat <- as.character(submission$FeatureName[i]) 
    # required as.character() ! 
    # Internal factor representations will not match corectly !!
    v <- predictions[imId,which(d.lookup==feat)]
    v <- min(96,1)
    v<- max(1,v)
    submission$Location[i] <- v   }
  
  submission$ImageId <-  NULL
  submission$FeatureName <- NULL
  sfile <-format(Sys.time(), "Submission%d%b%Yat%Hh%Mm%Ss.csv")
  write.csv(submission,file=sfile, row.names = FALSE, quote = FALSE)
  message("The file ",sfile," was saved for submission")
}