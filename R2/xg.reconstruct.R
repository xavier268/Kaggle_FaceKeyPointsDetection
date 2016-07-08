# =============================================================
# reconstruct key points from mouth center, lefteye, right eye
# =============================================================

if(!exists("df.train")) { source("xg.loadAll.R")}

# First, we construct models to predict each feature 
# from both eyes & mouth center (nose unreliable)

XX <- tbl_df(d.train) %>%
  select(
    mouth_center_x,
    mouth_center_y,
    right_eye_center_x,
    right_eye_center_y,
    left_eye_center_x,
    left_eye_center_y
  ) %>% mutate(intersect = 1)

mdl <- matrix(NA, ncol(d.train),7)
colnames(mdl) <- colnames(XX)

for(i in 1:ncol(d.train)) {
  c <- colnames(d.train)[i]
  message(i, "-Building model for ",c)
  sel <- !is.na(d.train[,i]) & !is.na(rowSums(XX))
  Y <- as.matrix(tbl_df(d.train)[sel,i])
  X <- as.matrix(XX[sel,])
  mdl[i,] <-   lm.fit(X,Y)$coefficients
}

rm(sel,i,c,X,XX,Y)

# Now, we define a function to reconstruct the data points from a partial set
#
# data : a 1x6 data frame, where the  mouth_center_x,
#     mouth_center_y,right_eye_center_x,right_eye_center_y,
#     left_eye_center_x,left_eye_center_y are defined
#     (instersect will be added). Data columns order is not relevant. 
#     All other columns are ignored.
#
# Value : a data frame row in the same format as d.train

xg.reconstruct <- function(data) {
  
  data <- tbl_df(data) %>% 
    select(
      mouth_center_x,
      mouth_center_y,
      right_eye_center_x,
      right_eye_center_y,
      left_eye_center_x,
      left_eye_center_y
    ) %>% mutate(intersect = 1)
  r<-  mdl %*%t(data)
  rownames(r) <- colnames(d.train)
  return (t(r))
}




