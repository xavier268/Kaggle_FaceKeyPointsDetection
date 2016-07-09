# build model to predict a set of given keypoint, 
# training only with those images where that keypoint is fully defined
# This version will try to compute several keypoints simultaneously

# unsing convolution (CNN) network

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


## --------------------------------------------------------------------
# define model version
m.version <- 2
buildModel <- function(kp) {
  # input : kp = a vector of int (1:30), to identify the feature we want to learn
  
  data <- mx.symbol.Variable("data")
  
  rdata <- mx.symbol.Reshape(data, shape = c(96,96,1,-1))
  
  fc1  <- mx.symbol.Convolution(data = rdata,  kernel=c(5,5),num.filter=50)
  act1 <- mx.symbol.Activation(data = fc1, act_type="relu")
  
  pl1 <- mx.symbol.Pooling(data=act1, pool.type = "max", kernel=c(2,2), stride=c(1,1)  )
  
  # fc2  <- mx.symbol.Convolution(data = pl1,  kernel=c(3,3),num.filter=30)
  # act2 <- mx.symbol.Activation(data = fc2, act_type="relu")
  # 
  # pl2<- mx.symbol.Pooling(data=act2, pool.type = "max", kernel=c(2,2), stride=c(2,2)  )
  
  ff <- mx.symbol.Flatten(data=pl1) ## needed after convolution steps ...
  
  fc1 <- mx.symbol.FullyConnected(ff, num_hidden = 120)
  act1 <- mx.symbol.Activation(fc1,act.type="relu")
  
  fc2 <- mx.symbol.FullyConnected(act1, num_hidden = 60)
  act2 <- mx.symbol.Activation(fc2,act.type="relu")
  
  fc3 <- mx.symbol.FullyConnected(act2, num_hidden = length(kp))
 
  output <- mx.symbol.LinearRegressionOutput(fc3, name = "output")
  
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
  print("Starting to train model for keypoints : ")
  print(kp)
  print(date())
  
  model <- mx.model.FeedForward.create(
    symbol = output,
    X = X,
    y = y,
    ctx = devices,
    num.round = 15,
    array.batch.size = 64,
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
## segment features
## ---------------------------------

print("Buiding/loading models")
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
if(file.exists("m2Model-symbol.json")) {
  print(paste0("Loading saved model 2, version ",m.version))
  m2 <- mx.model.load("m2Model",m.version)
} else {
  m2 <- buildModel(f2)
  mx.model.save(m2,prefix = "m2Model",m.version)
}





## --------------------------------
##  Compute full predictions with m2 (less precise)
## --------------------------------

print("Computing predictions")

x <- t(data.matrix(d.test[,-1]))
preds <- predict(m2,x,array.layout="colmajor")
colnames(preds) <- 1:1783
rownames(preds) <- (colnames(d.train[,1:30]))


# --------------------------------
## Now, we overwite the core 8 kp with with m1 (more precise)
# --------------------------------

preds1 <- predict(m1,x,array.layout="colmajor")
preds[f1,] <- preds1[1:8,]

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
          file = paste0("SubmissiongNmodel-v",m.version,"-",date(),".csv"), 
          quote = FALSE, 
          row.names = FALSE
)
print("Done.")
print(date())
