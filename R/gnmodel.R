# build model to predict a set of given keypoint, 
# training only with those images where that keypoint is fully defined
# This version will try to compute several keypoints simultaneously

# 
print(date())
require(mxnet)
if(!exists("d.train")) { source("loadData.R") }


## --------------------------------------------------------------------
buildModel <- function(kp) {
  # input : kp = a vector of int (1:30), to identify the feature we want to learn
  
  data <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data, name = "fc1", num_hidden = 30)
  act1 <- mx.symbol.Activation(fc1,act.type="relu")
  fc2 <- mx.symbol.FullyConnected(act1, name = "fc2", num_hidden = length(kp))
  output <- mx.symbol.LinearRegressionOutput(fc2, name = "output")
  
  devices <- mx.cpu()   # using 1 CPU
  mx.set.seed(42)
  
  # Format training data
  y <- d.train[,kp]
  X <- d.train[!is.na(rowSums(y)),-(1:30)]
  y <- d.train[!is.na(rowSums(y)),kp]
  
  # then convert to matrix/array, observations per column
  y <- t(data.matrix(y))
  X <- t(data.matrix(X))
  
  
  
  # Train the model with the training data
  tic <- Sys.time()
  print(paste("Starting to train model for keypoint : ", kp))
  print(date())
  
  model <- mx.model.FeedForward.create(
    symbol = output,
    X = X,
    y = y,
    ctx = devices,
    num.round = 3,
    array.batch.size = 100,
    learning.rate = 0.001,
    momentum = 0.9,
    eval.metric = mx.metric.rmse,
    initializer = mx.init.uniform(0.07),
    epoch.end.callback =  mx.callback.log.train.metric(10),
    array.layout = "colmajor"
  )
  print(paste("Finished training model for kp = ",kp))
  print(difftime(Sys.time(), tic))
  return (model)
}
## ---------------------------------------------------------------------------

# ## Test training on the first 2 keypoints
# 
# mm <- buildModel(1:5)
# x <- 1:9216
# dim(x) <- c(9216,1)
# p <- predict(mm,x,array.layout="colmajor")
# message("Prediction available in p")

## ---------------------------------
## segment features
## ---------------------------------
f1 <- c(1,2,3,4,21,22,29,30)  # frequently provided
f2 <- 1:30                    # full range

m1 <- buildModel(f1)
m2 <- buildModel(f2)

## --------------------------------
##  Compute full predictions with m2 (less precise)
## --------------------------------


x <- t(data.matrix(d.test[,-1]))
preds <- predict(m2,x,array.layout="colmajor")
colnames(preds) <- 1:1783
rownames(preds) <- (colnames(d.train[,1:30]))

## ---------------------------------
#  selction of the submission images where m1 or m2 model is needed
## ---------------------------------

sel2 <- d.lookup$FeatureName %in% colnames(d.train[0,1:30])[-f1]
sel2 <- unique(d.lookup$ImageId[sel2])

sel1 <- (1:(max(d.lookup$ImageId)))[-sel2]

# --------------------------------
## Now, we overwite only 8 kp with with m1 (more precise)
# --------------------------------

preds1 <- predict(m1,x,array.layout="colmajor")
preds[f1,sel1] <- preds1[f1,sel1]


## --------------
# fill up submission
## --------------
print("Formating submission")

for(i in 1:nrow(d.lookup)){
  d.lookup$Location <- preds[d.lookup$FeatureName,d.lookup$ImageId]
  if(i%%10 == 1) { message (i,"/",nrow(d.lookup))}
}

print("Saving predictions")
s <- tbl_df(d.lookup) %>% transmute(RowId = RowId, Location = Location)
write.csv(x=s, 
          file = paste0("SubmissiongNmodel",date(),".csv"), 
          quote = FALSE, 
          row.names = FALSE
)
print("Done.")
print(date())
