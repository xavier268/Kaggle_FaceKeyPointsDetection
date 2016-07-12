# ======================================
#
# This model will create two models, one to compute the 8 frequent keypoints,
# the second one, less precise, because less keypoints, for the rest.

# we use 2 kernels in paralele, a large one to capture face position,
# a smaller one to capture features.
#
# ======================================



## --------------------------------------------------------------------
# define model version
model.version <- 2
print(paste0("Model version : ", model.version), quote = FALSE)


#' Title
#'
#' @param kp : vector of int, describing the feature set to consider
#'
#' @return
#' @export
#'
#' @examples
buildModel <- function(kp, modeltype = 1) {
  if (modeltype == 1) {
    # model type 1
    data <- mx.symbol.Variable("data")
    
    fc1 <- mx.symbol.FullyConnected(data, num_hidden = 200)
    act1 <- mx.symbol.Activation(fc1, act.type = "relu")
    
    fc2 <- mx.symbol.FullyConnected(act1, num_hidden = 50)
    act2 <- mx.symbol.Activation(fc2, act.type = "relu")
    
    fc3 <- mx.symbol.FullyConnected(act1, num_hidden = length(kp))
    
    output <- mx.symbol.LinearRegressionOutput(fc3, name = "output")
    
  } else {
    # Model type 2
    
    data <- mx.symbol.Variable("data")
    
    rdata <- mx.symbol.Reshape(data, shape = c(96, 96, 1, -1))
    
    fc1  <-
      mx.symbol.Convolution(data = rdata,
                            kernel = c(5, 5),
                            num.filter = 4)
    act1 <- mx.symbol.Activation(data = fc1, act_type = "relu")
    pl1 <-
      mx.symbol.Pooling(
        data = act1,
        pool.type = "max",
        kernel = c(2, 2),
        stride = c(1, 1)
      )
    
    ff1 <- mx.symbol.Flatten(data = pl1)
    
    fc2  <-
      mx.symbol.Convolution(data = rdata,
                            kernel = c(30, 30),
                            num.filter = 1)
    act2 <- mx.symbol.Activation(data = fc2, act_type = "relu")
    pl2 <-
      mx.symbol.Pooling(
        data = act2,
        pool.type = "max",
        kernel = c(2, 2),
        stride = c(1, 1)
      )
    
    ff2 <- mx.symbol.Flatten(data = pl2)
    
    cc <- mx.symbol.Concat(list(ff1, ff2), num.args = 2)
    ff <- mx.symbol.Flatten(cc)
    
    fc1 <- mx.symbol.FullyConnected(ff, num_hidden = 60)
    act1 <- mx.symbol.Activation(fc1, act.type = "relu")
    
    # fc2 <- mx.symbol.FullyConnected(act1, num_hidden = 40)
    # act2 <- mx.symbol.Activation(fc2,act.type="relu")
    
    fc3 <- mx.symbol.FullyConnected(act1, num_hidden = length(kp))
    
    output <- mx.symbol.LinearRegressionOutput(fc3, name = "output")
    
    
    
    
    
  }
  
  devices <- mx.cpu()   # using 1 CPU
  
  # Format training data
  y <- d.train[, kp]
  X <- d.train[!is.na(rowSums(y)), -(1:30)]
  y <- d.train[!is.na(rowSums(y)), kp]
  
  # extract sample for cross validation
  if (model.eval) {
    e <- sample.int(nrow(X), model.eval)
    ye <- y[e,]
    Xe <- X[e,]
    y <- y[-e,]
    X <- X[-e,]
  }
  
  # then convert to matrix/array, observations per column
  y <- t(data.matrix(y))
  X <- t(data.matrix(X))
  eval.data <- NULL
  if (model.eval) {
    ye <- t(data.matrix(ye))
    Xe <- t(data.matrix(Xe))
    eval.data <- list(data = Xe, label = ye)
  }
  
  
  
  # Train the model with the training data
  tic <- Sys.time()
  print("Starting to train model for keypoints : ", quote = FALSE)
  print(kp)
  print(date())
  
  model <- mx.model.FeedForward.create(
    symbol = output,
    X = X,
    y = y,
    eval.data = eval.data,
    ctx = devices,
    num.round = model.round,
    array.batch.size = 64,
    learning.rate = model.lr,
    momentum = 0.9,
    eval.metric = mx.metric.rmse,
    initializer = mx.init.uniform(0.07),
    epoch.end.callback =  mx.callback.log.train.metric(10),
    array.layout = "colmajor"
  )
  print("Finished training model for kp : ", quote = FALSE)
  print(kp)
  print(difftime(Sys.time(), tic))
  return (model)
}


## ---------------------------------
## segment features
## ---------------------------------

print("Segmenting features", quote = FALSE)
f1 <- c(1, 2, 3, 4, 21, 22, 29, 30)  # frequently provided, more reliable
f2 <- 1:30                    # full range

## --------------------------------
##  Compute full predictions with m2 (less precise)
## --------------------------------

print("Computing predictions #2", quote = FALSE)

mx.ctx.default(mx.cpu())

if (file.exists(paste0(model.name, "_", 2, "-symbol.json"))) {
  print(paste0(
    "Loading saved model #2 for ",
    model.name,
    ", version ",
    model.version
  ))
  m2 <- mx.model.load(paste0(model.name, "_", 2), model.version)
} else {
  m2 <- buildModel(f2, 2)
  mx.model.save(m2, prefix = paste0(model.name, "_", 2), model.version)
}

x <- t(data.matrix(d.test[, -1]))
preds <- predict(m2, x, array.layout = "colmajor")
colnames(preds) <- 1:1783
rownames(preds) <- (colnames(d.train[, 1:30]))
rm(m2)
gc()

# --------------------------------
## Now, we overwite the core 8 kp with with m1 (more precise)
# --------------------------------

print("Computing predictions #1", quote = FALSE)

mx.ctx.default(mx.cpu())

if (file.exists(paste0(model.name, "_", 1, "-symbol.json"))) {
  print(paste0(
    "Loading saved model #1 for ",
    model.name,
    ", version ",
    model.version
  ))
  m1 <- mx.model.load(paste0(model.name, "_", 1), model.version)
} else {
  m1 <- buildModel(f1, 1)
  mx.model.save(m1, prefix = paste0(model.name, "_", 1), model.version)
}

preds1 <- predict(m1, x, array.layout = "colmajor")
preds[f1, ] <- preds1[1:8, ]
rm(m1)
gc()

# ensure predictions are between 0 and 1
preds <-
  apply(
    preds,
    FUN = function(x) {
      min(96, max(1, x))
    },
    MARGIN = c(1, 2)
  )

preds <- as.data.frame(preds)