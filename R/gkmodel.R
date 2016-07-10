# build model to predict a set of given keypoint, 
# training only with those images where that keypoint is fully defined
# This version will try to compute several keypoints simultaneously

# It computes the 8 most known kp, then extrapolates the others.

# using deep learning netwkork (DNN)

# 
print(date())
require(mxnet)
require(dplyr)
require(tidyr)

if (!exists("d.train")) {
  if (file.exists("loadData.RData")) {
    print("Loading compressed data")
    load("loadData.RData")
  } else {
    print("Generating data")
    source("loadData.R")
  }
  
}
print("Loading keypoint interpolation tool")
if(!exists("coefMatrix")) { source("kpmodel.R")}

## --------------------------------------------------------------------
# define model version
m.version <- 10


#' Compute and train a model to learn a subset of key points.
#'
#' @param kp : kp = a vector of int (1:30), to identify the feature we want to learn
#'
#' @return a trained model
#' @export
#'
#' @examples
buildModel <- function(kp) {
  data <- mx.symbol.Variable("data")
  
  # bdata <- mx.symbol.BatchNorm(data)
  
  rdata <- mx.symbol.Reshape(data, shape = c(96*96,-1))
  
  fc1 <- mx.symbol.FullyConnected(rdata, num_hidden = 200)
  act1 <- mx.symbol.Activation(fc1,act.type="relu")
  
  fc2 <- mx.symbol.FullyConnected(act1, num_hidden = 30)
  act2 <- mx.symbol.Activation(fc2,act.type="relu")
  
  fc3 <- mx.symbol.FullyConnected(act2, num_hidden = length(kp))
 
  output <- mx.symbol.LinearRegressionOutput(fc3, name = "output")
  
  devices <- mx.cpu()   # using 1 CPU
  mx.set.seed(42)
  
  # Format training data
  y <- d.train[,kp]
  X <- d.train[!is.na(rowSums(y)),-(1:30)]
  y <- d.train[!is.na(rowSums(y)),kp]
  
  # extract sample for cross validation
  e <- sample.int(nrow(X),500)
  ye <- y[e,]
  Xe <- X[e,]
  y <- y[-e,]
  X<- X[-e,]
  
  # then convert to matrix/array, observations per column
  y <- t(data.matrix(y))
  X <- t(data.matrix(X))
  ye <- t(data.matrix(ye))
  Xe <- t(data.matrix(Xe))
  
  
  # Train the model with the training data
  tic <- Sys.time()
  print("Starting to train model for keypoints : ")
  print(kp)
  print(date())
  
  model <- mx.model.FeedForward.create(
    symbol = output,
    X = X,
    y = y,
    eval.data = list(data=Xe, label=ye),
    ctx = devices,
    num.round = 500,
    array.batch.size = 200,
    learning.rate = 0.001,
    momentum = 0.9,
    eval.metric = mx.metric.rmse,
    initializer = mx.init.uniform(0.07),
    epoch.end.callback =  mx.callback.log.train.metric(10),
    array.layout = "colmajor"
  )
  print("Finished training model for kp : ")
  print(kp)
  print(difftime(Sys.time(), tic))
  return (model)
}


## ---------------------------------
## Build single model
## ---------------------------------

print("Buiding/loading only one model")
f1 <- c(1,2,3,4,21,22,29,30)  # frequently provided, more reliable
f2 <- 1:30                    # full range

mx.ctx.default(mx.cpu())

if(file.exists("m1Model-symbol.json")) {
  print(paste0("Loading saved model 1, version ",m.version))
  m1 <- mx.model.load("m1Model",m.version)
} else {
  m1 <- buildModel(f1)
  mx.model.save(m1,prefix = "m1Model",m.version)
}






## --------------------------------
##  Compute partial predictions with m1 (precise)
## --------------------------------

print("Computing predictions")

x <- t(data.matrix(d.test[,-1]))

preds1 <- t(predict(m1,x,array.layout="colmajor")) # n x 8
preds  <- t(predictkp(preds1)) # 30 x n
colnames(preds) <- 1:1783
rownames(preds) <- (colnames(d.train[,1:30]))



# ensure predictions are between 0 and 1
preds <- apply(preds, FUN = function(x) {min(96,max(1,x))}, MARGIN = c(1,2))

## --------------
# format submission
## --------------
print("Formating submission")

p <- tbl_df(preds) %>% 
  mutate(FeatureName = colnames(d.train[1,1:30]))%>%
  gather(key=ImageId, val=prediction, -FeatureName) %>%
  mutate(ImageId = as.integer(ImageId)) 

s <- d.lookup %>% 
  left_join(p, type="left", match="first",by = c("ImageId", "FeatureName")) %>%
  transmute(RowId = RowId, Location = prediction)

# Saving predictions

print("Saving predictions")
write.csv(x=s, 
          file = paste0("SubmissiongKmodel-v",m.version,"-",date(),".csv"), 
          quote = FALSE, 
          row.names = FALSE
)
print("Done.")
print(date())
