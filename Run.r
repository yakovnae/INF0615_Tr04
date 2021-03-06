# VERSION6 YAKOV
##########################################
# INF-0615 - Machine Learning            #
# Final Exercise - Digit Classification	 #
# Nomes: Ygor Pereira                    #
#        Yakov Nae                       #
##########################################


#---> no my data work samples ja soltar o 80% e 20%. chamar o my_data_partition de la mesmo
rm(list=ls())
library(neuralnet)
library(ggplot2)
rm(list=ls())
source("data_path.r")
source("Functions.r")
IPD<-1000
NITER<-5
XAXIS<-c(25,26,27,28,29,30,31)


data_raw <- read.csv(data_path, header=FALSE)
data_pca <- data_raw[,2:785]
data_pca <- data_pca[ , apply(data_pca, 2, var) != 0]

data_pca <- prcomp(data_pca ,
                   center = TRUE,
                   scale. = TRUE) 

pv <- data_pca$sdev^2/sum(data_pca$sdev^2)
print("PCA Importance:")
print(pv[1:20])
data_pca_selected <- data_pca$x[,1:15]
data_pca_selected <- cbind(data_raw[,1],data_pca_selected)
tmp  <- my_data_partition(data_pca_selected,0.6)
data_train <- tmp[[1]]; data_valid <- tmp[[2]]

hid<-lapply(XAXIS,function(x) c(5,x,3))
tmp <- NN_TEST1(data_train,data_valid,hid=hid,iPd=IPD,nIter=NITER,xticks=XAXIS,xlabel="c(5,x,3)")
tmp[[1]]
pdf('fig/TEST1_2.pdf'); tmp[[1]]; dev.off()

hid<-lapply(XAXIS,function(x) c(3,x))
tmp <- NN_TEST1(data_train,data_valid,hid,iPd=IPD,nIter=NITER,xticks=XAXIS,xlabel="c(3,x)")
tmp[[1]]
pdf('fig/TEST1_3.pdf'); tmp[[1]]; dev.off()
