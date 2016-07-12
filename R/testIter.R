# Testing use of Iterators and custum iterators for data augmentation
require("dplyr")
require("tidyr")
require("mxnet")


# define dummy, simple, stupid model
data <- mx.symbol.Variable("data")

fc3 <- mx.symbol.FullyConnected(data, num_hidden = 2)
dummy <- mx.symbol.LinearRegressionOutput(fc3, name = "output")

devices <- mx.cpu()   # using 1 CPU

# Format training data, colMajor !
X <- matrix(101:300, 10, 20)  # 10 values, 20 occurences
y <- matrix(1:40, 2, 20)      #  2 output values , 20 occurences

# define shadow iterator
it <-
  mx.io.arrayiter(
    data = X,
    label = y,
    shuffle = FALSE,
    batch.size = 7
  )

# define myit as a wrapper around it
myit <- list()
# myit$finalize <- function() {
#   stop("Finalize should not be called on myit !")
# }
# myit$initialize <- function() {
#   stop("Initialize should not be called on myit !")
# }
myit$iter.next <- function() {
  print("Hi there - inter.next !")
  it$iter.next()
}
myit$num.pad <- function() {
  print("Hi there - num.pad !")
  it$num.pad()
}
myit$reset <- function() {
  print("Hi there - reset !")
  it$reset()
}

myit$value <- function() {
  #print("Hi there - value !")
  #print("Transforming values for y")
  i <- it$value()
  #message("dim of x batch : ", paste(dim(i$data)," "))
  #message("dim of y batch : ", paste(dim(i$label)," "))
  i$label <- i$label
  i$data <- i$data - 100
  print("Iterated mini batch")
  print(i$label)
  print(i$data)
  return (i)
}
# make myit an S3 object to be recognized by Feedfrorward.create ...
# class(myit) <- append(class(myit),"Rcpp_MXArrayDataIter") # also a list
class(myit) <- "Rcpp_MXArrayDataIter" # just an iterator



message("Is it an S4 object : ", isS4(it))
message("Is myit an S4 object : ", isS4(myit))



# launch training
message(
  "Training 2 round, batch size = 7 ,  total size = 20 => 3 batches generated, padding the last batch !"
)
mx.model.FeedForward.create(
  dummy,
  X = myit,
  # Use the wrapper iterator !
  ctx = devices,
  learning.rate = 0.001,
  eval.metric = mx.metric.rmse,
  array.layout = "colmajor",
  num.round = 2,
  array.batch.size = 30
)
