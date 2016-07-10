# test convolutions

# see : https://github.com/dmlc/mxnet/issues/2138
# see : http://tjo-en.hatenablog.com/entry/2016/03/30/233848

require(mxnet)
require(dplyr)
require(tidyr)


# define a model with 1 linear regression outputs
data <-  mx.symbol.Variable('data')

fc1  <- mx.symbol.Convolution(data = data,  kernel=c(2,2),num.filter=5)
act1 <- mx.symbol.Activation(data = fc1, act_type="relu")
pl1 <- mx.symbol.Pooling(data=act1, pool.type = "max", kernel=c(2,2), stride=c(2,2)  )

ff <- mx.symbol.Flatten(data=pl1) ## needed after convolution steps ...

fc2  <- mx.symbol.FullyConnected(data = ff,  num_hidden=1)
net  <- mx.symbol.LinearRegressionOutput(data=fc2)


# Create artificial data - careful : colMajor convention, not the usual rowMajor !!
X <- array(1, c(96,96,1, 2140)) # we need 4 dimensions to enter the convolution layer !
y <-array(0.5,c(2140))



# create train/eval  subset for cross validation
eval <- sample.int(length(y), 300) # save 300 values for evaluation
# train
# caution we loose the articficial 3rd dimension, 
# moving from 4D 96x96x1x ... to 3D 96x96x ..
train.X <- X[,,,-(eval)]
train.y <- y[-eval]
# eval
eval.X <- X[,,,eval]
eval.y <- y[eval]
# Important : reshape the lost dimension
dim(train.X) <- c(96,96,1,dim(train.X)[3])
dim(eval.X) <- c(96,96,1,dim(eval.X)[3])

# =======================
# Initial training
# =======================

# train model
model <- mx.model.FeedForward.create (
  X = train.X,
  y = train.y,
  eval.data = list(data=eval.X, label=eval.y),
  eval.metric = mx.metric.rmse,
  ctx  = mx.cpu(),
  symbol = net,
  num.round  = 2,
  learning.rate  = 0.001,
  momentum = 0.9,
  wd = 0.00001,
  initializer  = mx.init.Xavier(),
  verbose = TRUE,
  epoch.end.callback = mx.callback.save.checkpoint("testCNNmodel"),
  array.layout = "colMajor"
)

# ==================================
# resume training from checkpoints
# ==================================
print("Continuing from saved model")

ll <- mx.model.load("testCNNmodel",2)

m2 <- mx.model.FeedForward.create (
    arg.params = ll$arg.params,
    symbol = ll$symbol,
    aux.params = ll$aux.params,
    
    num.round  = 4,
    X = train.X,
    y = train.y,
    eval.data = list(data=eval.X, label=eval.y),
    eval.metric = mx.metric.rmse,
    ctx  = mx.cpu(),
    learning.rate  = 0.001,
    momentum = 0.9,
    wd = 0.00001,
    # initializer  = mx.init.Xavier() - NO MORE INITIALIZER !
    verbose = TRUE,
    epoch.end.callback = mx.callback.save.checkpoint("testCNNmodel"),
    array.layout = "colMajor"
  )