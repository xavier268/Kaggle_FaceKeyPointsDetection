require(dplyr)
require(tidyr)


# Loading the files ----------

print("starting to load the files ...")
print(date())
train.file <- "../Dataset/training.csv"
test.file  <- "../Dataset/test.csv"
lookup.file <- "../Dataset/IdLookupTable.csv"
submission.file <- "../Dataset/SampleSubmission.csv"

d.train <- read.csv(train.file, stringsAsFactors=FALSE)
d.test <- read.csv(test.file, stringsAsFactors=FALSE)
d.lookup <- read.csv(lookup.file, stringsAsFactors=FALSE)
d.submission <- read.csv(submission.file,stringsAsFactors = FALSE)

rm(train.file,test.file,lookup.file, submission.file)

# Formating  data
print("starting to format data ...")
d.train <- tbl_df(d.train) %>%
  separate( col="Image", into=as.character(1:9216), convert=TRUE)
d.test <- tbl_df(d.test) %>%
  separate( col="Image", into=as.character(1:9216), convert=TRUE)
d.lookup <- tbl_df(d.lookup)

d.train[,-(1:30)] <- d.train[,-(1:30)]/255
d.test[,-1] <- d.test[,-1]/255

print("Saving compressed data")
save(d.lookup, d.test,d.train,d.submission, file="loadData.RData")

print("Done with data preparation")
print(date())







