#=======================
# Compute mean subimage from a df.train selection
#=======================

# sel : selection of rows from df.train. Firts 3 collumns are used, order is expected.
# s : size of subimage
# 
# value : an "averaged" subimage

source("xg.subimages.R")

xg.subimage.mean <- function(sel, s=5) {
  colMeans(xg.subimages(sel[,1:3],s),na.rm = TRUE)
}