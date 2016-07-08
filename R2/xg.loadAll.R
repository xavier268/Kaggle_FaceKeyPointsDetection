print("Loading compressed data")

library(dplyr)
library(tidyr)

load("d.train.RData")
load("df.train.RData")
load("im.train.RData")
load("im.test.RData")
set.seed(42)

print("Done.")