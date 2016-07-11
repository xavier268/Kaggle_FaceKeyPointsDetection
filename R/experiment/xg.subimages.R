
# =========================
# get subImages from a subset of df.train data
# =========================

source("xg.subimage.R")

# sel : a subset of df.train lines, only the first 3 columns are used (imid,x,y, in that order)
# s : subImage size
# 
# value : multiple rows of subImages
xg.subimages <- function(sel, s = 5) {
  t(apply(
    sel[,1:3],
    1,
    FUN = function(p) {
      # message(p," ",p[1], "  ", p[2],"  ",p[3])
      xg.subimage(
        im.train[p[1],],
        p[2],
        p[3],
        s)
    }
  ))
}