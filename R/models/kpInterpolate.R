## Function will model the missing 30-8 kpoints from the 8 selected keypoints
f1 <- c(1:4,21,22,29,30)

# Build a model to predict all keypoints from the f1 set of keypoints.
# creates a 9 x 30  matrix object (using closure)
print("Computing linear regression for kp", quote=FALSE)
coefMatrix <- (function() {
  
   # Our learning sets. Rowmajor ...
  kx  <- filter(d.train[,f1], !is.na(rowSums(d.train))) %>% mutate(inter=1)
  ky <- d.train[,1:30] %>% filter(!is.na(rowSums(d.train)))
  
  print("Computing kp interpolation coefficients", quote=FALSE)
  
 data.matrix((lm.fit(x=data.matrix(kx), y=data.matrix(ky)))$coefficients)
 
})()

## Adjust coefMatrix for exactness of know data
coefMatrix[,f1] <- 0
for(i in 1:8) {
  coefMatrix[i,f1[i]] <- 1
}


# Fillup a record from the 8 selected kpoints matrix (n x 8)
# return a n x 30 result matrix

predictkp<- function ( x ) {
  x <- data.matrix(cbind(x,rep.int(1,nrow(x)))) # n x 9
  return ( x %*% coefMatrix )
} 


# compute the rmse on the training set
sel <- rowSums(d.train[,f1])
evalKP <- predictkp(d.train[!is.na(sel),f1])
evalKP <- sqrt(sum((evalKP - d.train[!is.na(sel),f1]) ^2)) / (length(c(evalKP)))
message("rmse on training set for kp linear inetrpolation : ", evalKP)
rm(evalKP,sel)


