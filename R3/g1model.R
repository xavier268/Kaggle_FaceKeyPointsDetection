# build model to predict a given keypoint, 
# training only with those images where that keypoint is fully defined
# This version computes each keypoint separately ...

# 

require(mxnet)
if(!exists("d.train")) { source("loadData.R") }


## --------------------------------------------------------------------
buildModel <- function(kp) {
  # input : kp = 1 to 30, to identify the feature we want to learn
  
  data <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data, name = "fc1", num_hidden = 30)
  act1 <- mx.symbol.Activation(fc1,act.type="tanh")
  fc2 <- mx.symbol.FullyConnected(act1, name = "fc2", num_hidden = 1)
  output <- mx.symbol.LinearRegressionOutput(fc2, name = "output")
  
  devices <- mx.cpu()   # using 1 CPU
  mx.set.seed(42)
  
  # Format training data
  y <- d.train[,kp]
  X <- d.train[!is.na(rowSums(y)),-(1:30)]
  y <- d.train[!is.na(rowSums(y)),kp]
  
  # then convert to matrix/array, observations per column
  y <- c(data.matrix(y))
  X <- t(data.matrix(X))
  
  
  
  # Train the model with the training data
  tic <- Sys.time()
  print(paste("Starting to train model for keypoint : ", kp))
  
  model <- mx.model.FeedForward.create(
    symbol = output,
    X = X,
    y = y,
    ctx = devices,
    num.round = 20,
    array.batch.size = 100,
    learning.rate = 0.000001,
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

## Build all the models, and save them


model <- list()
for(i in 1:30) {
  model[[i]] <- buildModel(i)
}

# -----------------------------------------------------------------------

## save predictions in d.lookup
computePredictions <- function() {
  for(i in 1:nrow(d.lookup)) {
    f <- as.character(d.lookup$FeatureName[i])
    kp <- which(colnames(d.train) ==f)
    im <- d.lookup$ImageId[i]
    r <- make1Prediction(im,kp)
    #message(r)
    d.lookup$Location[i] <- r
    if(i %% 10 == 1) { message(i,"/",nrow(d.lookup))}
    return (d.lookup)
  }
}
print("Computing predictions")
d.lookup <- computePredictions()

print("Saving predictions")
s <- tbl_df(d.lookup) %>% transmute(RowId = RowId, Location = Location)
write.csv(x=s, 
          file = paste0("Submissiong1model",date(),".csv"), 
          quote = FALSE, 
          row.names = FALSE
          )
print("Done.")
