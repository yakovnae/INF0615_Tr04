##########################################
# INF-0615 - Machine Learning            #
# Final Exercise - Digit Classification	 #
# Nomes: Ygor Pereira                    #
#        Yakov Nae                       #
##########################################
# load the kernlab package
#install.packages("kernlab")
#install.packages("e1071"")

library(kernlab)
set.seed(42)

# load the toy datasets
source("data_path.r")
data_all <- read.csv(data_path, header=FALSE)

#slit train and test data from total
rows <- nrow(data_all)
cols <- ncol(data_all)
ntrain <- round(rows*0.6) # number of training examples
tindex <- sample(rows,ntrain) # indices of training samples

data_train <- data_all[tindex,]
data_test <- data_all[-tindex,]



######## LINEAR-SVM ############
timestamp()
xtrain <- as.matrix(data_train[,2:cols])
ytrain <- as.factor(data_train[,1])

# Compute accuracy
calculateAccuracy <- function(svp, xdata, ytestdata, yprediction){
  
  sum(yprediction==ytestdata)/length(ytestdata)
 
  return(sum(yprediction==ytestdata)/length(ytestdata))
}
########## PREDICT WITH SVM #############

xtest <- data_test[,2:cols]
ytest <- data_test[,1] 


####### INFLUENCE OF C #############
CList = 10^seq(-5,-10)
bestC <- 0
bestAcc <- 0
bestSVP <- NULL
cs <- c()
accs <- c()

for (C in CList){
  svp <- ksvm(xtrain,ytrain,type="C-svc",kernel="vanilladot",C=C,scaled=c())
  
  ypred = predict(svp,xtest)
  
  print(as.matrix(table(Actual = ytest, Predicted = ypred)))
  
  acc <- calculateAccuracy(svp, xtest, ytest, ypred)
  
  print(paste(C, " - ", acc))
  
  cs <- c(cs, C)
  accs <- c(accs, acc)
  
  if(acc > bestAcc) {
    bestAcc <- acc
    bestC <- C
    bestSVP <- svp
  }
}

timestamp()

print(paste('Best C = ', bestC, " with accurary = ", bestAcc))

df <- data.frame(ACC=accs, C=cs)
print(df)


