# ========================
#  Compute "average" subimages for various parts
# ========================

if(!exists("df.train")) { source("xg.loadAll.R") }
source("xg.subimage.mean.R")

im.mean <- list()

# setting image radius
s<- 15
ss <- 2*s + 1
im.mean$s <- s

# mouth center
sel <- df.train %>%
  filter(!is.na(x), !is.na(y), part=="mouth_center") %>%
  select(imid,x,y)
im.mean$mouth <- xg.subimage.mean(sel, s)

# display average mouth
image(1:ss,1:ss, matrix(rev(im.mean$mouth),ss,ss),col=grey(0:255/255),asp=1.)
title(main="average mouth")

# nose tip
sel <- df.train %>%
  filter(!is.na(x), !is.na(y), part=="nose_tip") %>%
  select(imid,x,y)
im.mean$nose <- xg.subimage.mean(sel, s)

# display average nose
image(1:ss,1:ss, matrix(rev(im.mean$nose),ss,ss),col=grey(0:255/255),asp=1.)
title(main="average nose")

# left eye
sel <- df.train %>%
  filter(!is.na(x), !is.na(y), part=="eye_center", side=="left") %>%
  select(imid,x,y)
im.mean$leye <- xg.subimage.mean(sel, s)

# display average nose
image(1:ss,1:ss, matrix(rev(im.mean$leye),ss,ss),col=grey(0:255/255),asp=1.)
title(main="left eye")

# right  eye
sel <- df.train %>%
  filter(!is.na(x), !is.na(y), part=="eye_center", side=="right") %>%
  select(imid,x,y)
im.mean$reye <- xg.subimage.mean(sel, s)

# display average nose
image(1:ss,1:ss, matrix(rev(im.mean$reye),ss,ss),col=grey(0:255/255),asp=1.)
title(main="right eye")

# Average postion of key points
d.means <- colMeans(d.train, na.rm = TRUE)

rm(sel)
rm(ss)
rm(s)

