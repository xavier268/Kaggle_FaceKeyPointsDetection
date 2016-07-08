# ===========================
# Find a sub image that has the best correlation within an image
# ===========================

# im : a 96*96 image
# subim : a subimage, 2*s+1 x 2*s+1
# where : a matrix, where each row (x,y) reprensents the value to explore. 
# (Use expand.grid to create ...)
# debug : do we display the image matched ?
#
# value : a matrix, where each row represent (x, y, c) - c is the correlation, sorted by decreasing correlation

source("xg.subimage.R")


xg.subimage.find<-function(im, subim, where = NA, debug=FALSE) {
  
  subim <- c(subim)
  s <- sqrt((length(subim)))
  s <- round((s-1)/2)
  
  # message("Computed radius, s for subimage : ",s)
  
  if(is.na(where)) {
    where <- expand.grid((1+s):(96-s),(1+s):(96-s))
  }
  
  r <- as.tbl(where) %>%
    transmute(x=Var1, y=Var2) %>%
    mutate(c = NA)
  
  r <- mutate(r,c = apply(r,1,FUN=function(p) {
    i <- xg.subimage(im,p[1],p[2],s)
    cor(subim,c(i))
  })) %>%
    arrange(desc(c))
  
  if(debug) {
    image(1:96,1:96,matrix(im,96,96), col=grey(0:255/255),asp=1.)
    points(r$x[1],r$y[2], col="red", pch=3)
    title(main=paste("cor = ",r$c[1]))
  }
  return(r)
}
