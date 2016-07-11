# ======================
# Compute subImages
#=======================
# im : image (96*96, vector)
# x,y : location of center of sub image
# s : "radius" of square subImage, from x-s to x+s ...
# 
# Value : a subImage in vector format, length (2s+1)^2
# 
# Notes : clipping happens, when to close to limits. Non existing ilages are filled with an "average" value.
#         subImage is normalized netween 0 and 1

library(ripa)

xg.subimage<- function(im, x=48, y=48, s=5) {
  x <- round(x)
  y <- round(y)
  if(x-s<1 | x+s > 96 | y-s<1 | y+s > 96) { 
    # message("Wrong dimensions outside -> clipping  :",x," - ",y)
    
    bg <- matrix(128,96*3,96*3)                   # create background
    bg[97:192,97:192] <- matrix(im,96,96)         # copy im in background
    mm <- bg[(96+x-s):(96+x+s),(96+y-s):(96+y+s)] # extract from background
    return(normalize(c(mm)))
  }
  # message("im length : ",length(im)," x",x," y ",y)
  
  mm <- matrix(im,ncol=96,nrow=96)
  mm <- mm[(x-s):(x+s),(y-s):(y+s)] # the sub image ...
  #  message("Length mm", length(c(mm)), " dim mm ",dim(mm))
  r <- normalize(c(mm))
  # message("r length : ",length(r))
  return (r)
}
