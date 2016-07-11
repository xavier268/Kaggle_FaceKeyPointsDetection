#=========================
# find the best combination for the 2 eyes, mouth center and nose tip
#=========================

# im : an image
# 
# value : a list, specifying nose, mouth, leye, reye positions and associated correlations
# sorted by decreasing likelyhood

if(!exists("im.mean")) {source("xg.compute.means.R")}
source("xg.subimage.find.R")

xg.subimage.find.means <- function(im, debug=FALSE) {
  
  s <- im.mean$s
  ss <- 2*s+1
  
  leye <- xg.subimage.find(im,im.mean$leye)
  reye <- xg.subimage.find(im,im.mean$reye)
  # nose <- xg.subimage.find(im,im.mean$nose)
  mouth <- xg.subimage.find(im,im.mean$mouth)
  
  # ------- eliminate grossly invalid combinations --------
  
  leye <- filter(leye, 
                 abs(x-d.means["left_eye_center_x"]) < 30,
                 abs(y-d.means["left_eye_center_y"]) < 30,
                 x > 48, 
                 y < 60)
  reye <- filter(reye, 
                 abs(x-d.means["right_eye_center_x"]) < 30,
                 abs(y-d.means["right_eye_center_y"]) < 30,
                 x < 48, 
                 y < 60)
  mouth <- filter(mouth,
                  abs(x-d.means["mouth_center_x"]) < 40,
                  abs(y-d.means["mouth_center_y"]) < 40,
                  y > 50
                  )
  # nose<- filter(nose, y <80, y > 20,x<70, x> 30)
  # Nose looks out of sync very often !
  


  r <- list()
  r$leye <- leye[1,]
  r$reye <- reye[1,]
  
  message("Mouth pos before ",nrow(mouth))
  # force mouth to be 20 above average eyes
  mouth <- filter(mouth, 
                  mouth$y*2 > 40 + reye$y[1] + leye$y[1])
  
  # force mouth to be aligned with the eyes ...
  # --------------------------------------------
  delta_dist <- function(x,y) {
   return (abs( (leye[[1, 1]] - x) ^ 2
    +(leye[[1, 2]]- y) ^ 2
    -(reye[[1, 1]] - x) ^ 2
    -(reye[[1, 2]] - y) ^ 2
    ))
  } # -------------------------------------------
  
  
  mouth_min <- min(delta_dist(mouth$x,mouth$y)) * 3+40
  message("mouth_min : ",mouth_min)
  mouth <- filter(mouth, 
                 delta_dist(mouth$x,mouth$y)<=mouth_min)
  
  message("Mouth pos after ",nrow(mouth))
  r$mouth <- mouth[1,]
  #r$nose <- nose[1,]
  
  # TODO : if cor < limit, then use d.means instead
  
  if(debug) {
    image(1:96,1:96,matrix(im,96,96), col=grey(0:255/255),asp=1.)
    points(leye$x[1],leye$y[1], col="red", pch=3)
    points(reye$x[1],reye$y[1], col="green", pch=3)
    points(mouth$x[1],mouth$y[1], col="blue", pch=3)
    #points(nose$x[1],nose$y[1], col="yellow", pch=3)
    title(main=paste("cor : ",(mouth$c[1] +leye$c[1] + reye$c[1])/3))
  }
                
  return (r)
  
}

