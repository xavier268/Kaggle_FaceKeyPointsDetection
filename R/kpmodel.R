## Function will model the missing 30-8 kpoints from the 8 selected keypoints
f1 <- c(1:4,21,22,29,30)

# Build a model to predict all keypoints from the f1 set of keypoints.
# creates a 9 x 30  matrix object (using closure)
coefMatrix <- (function() {
  
   # Our learning sets. Rowmajor ...
  kx  <- filter(d.train[,f1], !is.na(rowSums(d.train))) %>% mutate(inter=1)
  ky <- d.train[,1:30] %>% filter(!is.na(rowSums(d.train)))
  
  print("Computing kp interpolation coefficients")
  
 data.matrix((lm.fit(x=data.matrix(kx), y=data.matrix(ky)))$coefficients)
})()

# Fillup a record from the 8 selected kpoints matrix (n x 8)
# return a n x 30 result matrix

predictkp<- function ( x ) {
  x <- data.matrix(cbind(x,rep.int(1,nrow(x)))) # n x 9
  return ( x %*% coefMatrix )
} 
