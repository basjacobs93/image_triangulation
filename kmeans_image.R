library(imager)
library(triangular)
library(tidyverse)

im <- load.image("blauwborst.jpeg")

plot(im)

im_df <- im %>% 
  sRGBtoLab() %>% 
  as.data.frame(wide = "c")

km <- im_df %>% 
  select(-x, -y) %>% 
  kmeans(centers = 5)

im_clustered <- LabtosRGB(as.cimg(c(km$centers[km$cluster,]), dim = dim(im)))

plot(im_clustered)

sc.spat <- (dim(im)[1:2]*.28) %>% max #Scale of spatial dimensions
sc.col <- imsplit(im,"c") %>% map_dbl(sd) %>% max

# Source: https://dahtah.wordpress.com/2017/03/24/superpixels-in-imager/
slic <- function(im, nS, compactness=1,...) {
  #If image is in colour, convert to CIELAB
  if (spectrum(im) ==3) im <- sRGBtoLab(im)
  
  #The pixel coordinates vary over 1...width(im) and 1...height(im)
  #Pixel values can be over a widely different range
  #We need our features to have similar scales, so
  #we compute relative scales of spatial dimensions to colour dimensions
  sc.spat <- (dim(im)[1:2]*.28) %>% max #Scale of spatial dimensions
  sc.col <- imsplit(im,"c") %>% map_dbl(sd) %>% max
  
  #Scaling ratio for pixel values
  rat <- (sc.spat/sc.col)/(compactness*10)
  
  
  X <- as.data.frame(im*rat,wide="c") %>% as.matrix
  #Generate initial centers from a grid
  ind <- round(seq(1,nPix(im)/spectrum(im),l=nS))
  #Run k-means
  km <- kmeans(X,X[ind,],...)
  
  #Return segmentation as image (pixel values index cluster)
  seg <- as.cimg(km$cluster, dim=c(dim(im)[1:2],1,1))
  #Superpixel image: each pixel is given the colour of the superpixel it belongs to
  sp <- map(1:spectrum(im),~ km$centers[km$cluster,2+.]) %>% do.call(c,.) %>% as.cimg(dim=dim(im))
  #Correct for ratio
  sp <- sp/rat
  if (spectrum(im)==3)
  {
    #Convert back to RGB
    sp <- LabtosRGB(sp) 
  }
  list(km=km,seg=seg,sp=sp)
}

out <- slic(im, 100)

plot(out$sp)


