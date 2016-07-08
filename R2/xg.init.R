

# =============================
# This script loads the original **`r params$train`, `r params$test` and `r params$lookup`** files into memory. 
# As this requires some time, once loaded, it will save the related R object in specific RData environment files.
# =============================

library(dplyr)
library(tidyr)


# Loading the files ----------

print("starting to load the files ...")
train.file <- "../Dataset/training.csv"
test.file  <- "../Dataset/test.csv"
lookup.file <- "../Dataset/IdLookupTable.csv"

d.train <- read.csv(train.file, stringsAsFactors=FALSE)
d.test <- read.csv(test.file, stringsAsFactors=FALSE)
d.lookup <- read.csv(lookup.file, stringsAsFactors=FALSE)



#  Formatting the loaded data --------------
print("Starting to format files ...")
d.lookup$Location <- NULL
d.lookup$ImageId <- NULL
d.lookup$RowId <- NULL
d.lookup <- d.lookup[1:30,]

im.train      <- d.train$Image
d.train$Image <- NULL
im.test       <- d.test$Image
d.test$Image  <- NULL



# Single image
d.convertLoadedImage  <- function(image) {
  as.integer(unlist(strsplit(image," ")))
}


# Rows of images
d.convertLoadedImages <- function(images) {
  n <- length(images)
  res <- matrix(nrow =n,ncol = 9216)
  for (i in 1:n) {
      res[i,] <-d.convertLoadedImage(images[i])
      }
  return(res)
}

print("Starting to convert images ...")
im.train <- d.convertLoadedImages(im.train)
im.test <- d.convertLoadedImages(im.test)

# cleanup
rm (d.convertLoadedImage)
rm(d.convertLoadedImages)
rm(d.test)

# Adding a column for the "mouth_center"
d.train$mouth_center_x <- (d.train$mouth_center_top_lip_x + d.train$mouth_center_bottom_lip_x)/2
d.train$mouth_center_y <- (d.train$mouth_center_top_lip_y + d.train$mouth_center_bottom_lip_y)/2


# formatted test data - df.train
print("Creating formatted test data df.train")
df.train <- as.tbl(d.train) %>%
  mutate(imid = seq_along(d.train[,1])) %>%
  gather(key = "item" , value = "coord" , -imid ) %>%
  arrange(imid) %>%
  extract(item,
          into=c("side","part","xy"), 
          regex="^(?:(left|right)_)?([[:alnum:]_]+)_(x|y)$"
  ) %>%
  spread(xy,coord) %>%
  select(imid=as.numeric(imid),x=as.numeric(x),y=as.numeric(y),side,part)





# Saving the environment ------------------
print("Saving environment")
save(d.train,file="d.train.RData")
save(im.train, file="im.train.RData")
save(im.test,file="im.test.RData")
save(df.train, file="df.train.RData")

print("Done.")