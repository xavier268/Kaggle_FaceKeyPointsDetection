# ========================================
#
#  A Deep Neural Network configuration
#
#     We compute the 8 most frequent keypoints with the DNN
#     We then interpolate the other keypoinst using the kpInterpolate model
#
# ========================================

model.version <- 1
print(paste0("Model version : ",model.version), quote=FALSE)

#' Compute and train a NN model to learn a subset of key points.
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
  
  # Format training data
  y <- d.train[,kp]
  X <- d.train[!is.na(rowSums(y)),-(1:30)]
  y <- d.train[!is.na(rowSums(y)),kp]
  
  # extract sample for cross validation
  if (model.eval) {
    e <- sample.int(nrow(X), model.eval)
    ye <- y[e, ]
    Xe <- X[e, ]
    y <- y[-e, ]
    X <- X[-e, ]
  }
  
  # then convert to matrix/array, observations per column
  y <- t(data.matrix(y))
  X <- t(data.matrix(X))
  eval.data <- NULL
  if(model.eval) {
    ye <- t(data.matrix(ye))
    Xe <- t(data.matrix(Xe))
    eval.data <- list(data=Xe, label=ye)
  }
  
  
  # Train the model with the training data
  tic <- Sys.time()
  print("Starting to train model for keypoints : ", quote=FALSE)
  print(kp)
  print(date(), quote=FALSE)
  
  model <- mx.model.FeedForward.create(
    symbol = output,
    X = X,
    y = y,
    eval.data = eval.data,
    ctx = devices,
    num.round = model.round,
    array.batch.size = 200,
    learning.rate = model.lr,
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
## Build NN  model
## ---------------------------------

print("Buiding/loading only one model", quote=FALSE)
f1 <- c(1,2,3,4,21,22,29,30)  # frequently provided, more reliable

mx.ctx.default(mx.cpu())

if(file.exists(paste0(model.name,"-symbol.json"))) {
  print(paste0("Loading saved model: ",model.name," version: ",model.version))
  m1 <- mx.model.load(model.name,model.version)
} else {
  m1 <- buildModel(f1)
  mx.model.save(m1,prefix = model.name,model.version)
}





## -------------------------------
##    load kp interpolation tool
## -------------------------------

if(!exists("coefMatrix")) { source("models/kpInterpolate.R")}


## --------------------------------
##  Compute and return predictions 
## --------------------------------

print("Computing predictions", quote=FALSE)

x <- t(data.matrix(d.test[,-1]))

preds1 <- t(predict(m1,x,array.layout="colmajor")) # n x 8
preds  <- t(predictkp(preds1)) # 30 x n
colnames(preds) <- 1:1783
rownames(preds) <- (colnames(d.train[,1:30]))
rm(preds1)
preds <- as.data.frame(preds)
