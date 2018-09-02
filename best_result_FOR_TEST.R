##########################################
# INF-0615 - Machine Learning            #
# Final Exercise - Digit Classification	 #
# Nomes: Ygor Pereira                    #
#        Yakov Nae                       #
##########################################

# Este foi o melhor metodo para previsao
# Este arquivo deve ser rodado para o teste com o arquivo de teste final

timestamp()

#ARQUIVOS A SEREM USADOS
print("READ DATA")
#Treino
data_train <- read.csv("mnist_trainVal.csv", header=FALSE)

#Teste 
data_test <- read.csv("mnist_test.csv", header=FALSE)

library(kernlab)
set.seed(42)


######## LINEAR-SVM ############
cols <- ncol(data_train)
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

C <- 1e-06
print("CALCULATE SVM")
svp <- ksvm(xtrain,ytrain,type="C-svc",kernel="vanilladot",C=C,scaled=c())

print("PREDICT WITH SVM")
ypred = predict(svp,xtest)

print(as.matrix(table(Actual = ytest, Predicted = ypred)))

acc <- calculateAccuracy(svp, xtest, ytest, ypred)

print(paste("ACCURACY = ", (acc * 100), "%"))

timestamp()