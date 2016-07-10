# ===========================
#
#  Simple, stupid, model, that fills the prediction matrix 
#  with the means of the training keypoints
#
# ===========================

print(paste0("Entering model : ", model.name),quote=FALSE)

print("This model will compute the means of the training keypoints and produce a constant prediction with that value")

preds <- matrix(colMeans(d.train[,1:30], na.rm = TRUE), 
                30, 
                1783, 
                byrow = FALSE)

preds <- as.data.frame(preds)

print(paste0("Exiting model : ", model.name),quote=FALSE)