# VERSION6 YAKOV
##########################################
# INF-0615 - Machine Learning            #
# Final Exercise - Digit Classification  #
# Nomes: Ygor Pereira                    #
#        Yakov Nae                       #
##########################################
#TODO: Find out how to limit the number of iterations of main algorithm so it stops if not converges

# Função para plotar uma imagem a partir de um único feature vector
# imgArray = fv da imagem, com classe na V1 e os pixels em V2 ~ V785
# os pixels podem estar no intervalo [0.0,1.0] ou [0, 255]
plotImage = function(imgArray){
  # Transforma array em uma matrix 28x28 (ignorando a classe em V1)
  imgMatrix = matrix((imgArray[2:ncol(imgArray)]), nrow=28, ncol=28)
  
  # Transforma cada elemento em numeric
  im_numbers <- apply(imgMatrix, 2, as.numeric)
  
  # Girando a imagem apenas p/ o plot
  flippedImg = im_numbers[,28:1]
  
  image(1:28, 1:28, flippedImg, col=gray((0:255)/255), xlab="", ylab="")
  title(imgArray[1])
}

my_data_partition <- function(data,p){
    #Spliting the data matrix to train and validation
    # matrix by the proportion p.

    if ( (p<=0)||(p>1) ) return(NULL)
    n <- nrow(data)
    train_index <- my_data_work_samples(data,round(p*n))
    data_train  <- data[train_index,]
    
    data_valid  <- data[-train_index,]
    valid_index <- sample(1:nrow(data_valid),round((1-p)*n),replace=F)
    data_valid <- data_valid[valid_index,]
    return(list(data_train,data_valid))
}

my_nn <- function(data_train,data_valid,
                  hiddenV=c(5,3),iPd=100,nIter=10,seed_=42,method="NN",C=1){
  #iPd - images per digit
  set.seed(seed_)
  oneVSall_train <- matrix(0,dim(data_train)[1],10)
  colnames(oneVSall_train)<-c(0:9)
  oneVSall_valid <- matrix(0,dim(data_valid)[1],10)
  colnames(oneVSall_valid)<-c(0:9)
  ind_mat <- my_get_index_matrix(data_train)
  #input verification
  if (iPd>dim(ind_mat)[2]) {
    print(paste(format(Sys.time(), "%d-%m %X"),"iPd parameter is greater than images in data_train"))
    return(NULL)
  }
  
  nDigits <- 10 #10 digits
  pb <- txtProgressBar(min = 1, max = 10*nIter-1, style = 3)
  print(paste(format(Sys.time(), "%d-%m %X"),"MainFor:",iPd, "images per digit,",nIter,"iterations per digit"))
 
  for (j in 1:nIter){
    # create progress bar
    for (i in 1:nDigits){
      data_train2<-data_train[my_data_work_samples2(i-1,ind_mat,iPd),]
      colnames(data_train2) <- paste("v",1:dim(data_train2)[2],sep='')
      colnames(data_valid)  <- paste("v",1:dim(data_valid)[2], sep='')
      f <- my_get_formula(data_train2)
      num <- data_train2[,1]

      data_train2[num==(i-1),1] <- 1
      data_train2[num!=(i-1),1] <- 0
      tryCatch({
                if (method=="NN"){
                    nnModel = neuralnet(formula=f, data=data_train2, hidden=hiddenV, linear.output=FALSE,threshold = 0.05)
                }
                else{
                    nnModel = ksvm(data_train2[,2:ncol(data_train2)],num,type="C-svc",kernel="vanilladot",C=C,scaled=c())
                } 
        },
        warning = function(w) {
            print(paste("NN Did not converge for digit",i-1,"and iteration",j)) 
        }
      )
      if (method=="NN"){
        oneVSall_train[,i] <- t(my_nn_compute(nnModel, data_train[,2:dim(data_train)[2]]) )
        oneVSall_valid[,i] <- t(my_nn_compute(nnModel, data_valid[,2:dim(data_valid)[2]]) )
      }
      else{
        oneVSall_train[,i] <- predict(model,data)
        oneVSall_valid[,i] <- predict(model,data)
      } 
      
      setTxtProgressBar(pb, nDigits*(j-1)+i)
    }
  }
  close(pb)
  oneVSall_trainV <- max.col(oneVSall_train, 'random') - 1
  oneVSall_validV <- max.col(oneVSall_valid, 'random') - 1
  
  print(paste(format(Sys.time(), "%d-%m %X"),"MainFor:",iPd, "images per digit,",nIter,"iterations per digit"))
  print( paste( format(Sys.time(), "%d-%m %X"), " ", floor(sum(rowSums(oneVSall_train==oneVSall_trainV)!=0)/length(oneVSall_trainV)*100) ,"% withraws in TRAIN") )
  print( paste( format(Sys.time(), "%d-%m %X"), " ", floor(sum(rowSums(oneVSall_valid==oneVSall_validV)!=0)/length(oneVSall_validV)*100) ,"% withraws in VALID") )
  
  print( as.matrix(table(Actual = data_train[,1], Predicted = oneVSall_trainV)) )
  print( as.matrix(table(Actual = data_valid[,1], Predicted = oneVSall_validV)) )
  
  return(c( sum(data_train[,1]==oneVSall_trainV)/length(oneVSall_trainV),
            sum(data_valid[,1]==oneVSall_validV)/length(oneVSall_validV) ) )
}
my_nn_compute <- function(model,data){
  nnCompute  <- compute(model, data)
  prediction <- nnCompute$net.result
  th <- .5
  prediction[prediction <  th] = -1
  prediction[prediction >= th] = 1
  return(prediction)

#  return(round(prediction*100))  
}

my_get_formula <- function(data){
  feats <- colnames(data)
  
  # Concatena o nome de cada feature, ignorando a primeira
  f <- paste(feats[2:length(feats)],collapse=' + ')
  f <- paste(feats[1],' ~',f)
  
  # Converte para fórmula
  f <- as.formula(f)
  f
}

my_get_index_matrix <- function(data){
  #getting minimum size index vector
  min<-Inf
  for (i in 0:9){
    n<-length(which(data[,1] %in% i))
    if (n < min) min<-n
  }
  ind_mat <- matrix(0,10,n)
  for (i in 0:9){
    ind <- which(data[,1] %in% i)
    ind_mat[i+1,] <- ind[1:n]
  }
  return(ind_mat)
}

my_data_work_samples2 <- function(N,ind_mat,nlength){
  nlength <- floor(nlength/18)*18
  sampN <- sample(round(dim(ind_mat)[2]/2),round(nlength/2),replace = F)
  ind_out<-ind_mat[N+1,sampN]
  for (i in 0:9){
    if (i!=N){
      sampN   <- sample(round(dim(ind_mat)[2]/2),round(nlength/2/9),replace = F)
      ind_out <- c(ind_out,ind_mat[i+1,sampN])
    }
  }
  return(ind_out)
}
my_data_work_samples <- function(data,n){
  #Returns indexes of n/10 entries from each number 0-9
  ind_out<-c()
  for (i in 0:9){
    ind <- which(data[,1] %in% i)
    ind <- ind[sample(length(ind),round(n/10),replace = F)]
    ind_out <- c(ind_out,ind)
  }
  return(ind_out)
}

my_lineplot <- function(l,lables=paste("v",1:length(l),sep=''),xticks=1:length(l[[1]]),xlabel="",ylabel="",title=""){
  x_axis <- rep(xticks, times=length(l))
  types  <- rep(lables,each=length(l[[1]]))
  values <- unlist(l)
  d <- data.frame(x=x_axis,type=types,v=values)
  g <- ggplot(data=d, aes(x=x, y=v))
  g <- g + geom_line(aes(color=type))
  if (xlabel != "") g <- g + xlab(xlabel)
  if (ylabel != "") g <- g + ylab(ylabel)
  if (title!="") g <- g+ labs(title=title)
  return(g)
}

NN_TEST1 <- function(data_train,data_valid,hid,iPd=100,nIter=10,xticks=1:length(hiddenV),xlabel="c(?,x)"){
  trainE<-c()
  validE<-c()
  for (i in 1:length(hid)){
    t <- Sys.time();
    tmp <- my_nn(data_train,data_valid,hid[[i]],iPd=iPd,nIter=nIter)
    trainE <- c(trainE, 1 - tmp[1])  
    validE <- c(validE, 1 - tmp[2]) 
    Sys.time() - t
    print(paste(format(Sys.time(), "%d-%m %X")," round", i, "of", length(hid),"  errorT=",round(1 - tmp[1],2),"  errorV=", round(1 - tmp[2],2)) )
  }
  return( list( my_lineplot(list(trainE,validE),c("train","valid"),xlabel = xlabel,ylabel = "Error",xticks=xticks),
                list(trainE,validE) ) )  
}
my_normalize <- function(dtrain,dvalid){
  n <- ncol(dtrain)-1
  meanTrain <- colMeans(dtrain[,1:n]) #mean of each feature
  stdTrain  <- apply(dtrain[,1:n], 2, sd) #std of each feature
  
  dtrain[,1:n] <- sweep(dtrain[,1:n], 2, meanTrain, "-")
  dtrain[,1:n] <- sweep(dtrain[,1:n], 2, stdTrain,  "/")
  dvalid[,1:n] <- sweep(dvalid[,1:n], 2, meanTrain, "-")
  dvalid[,1:n] <- sweep(dvalid[,1:n], 2, stdTrain,  "/")
  
  return(list(dtrain,dvalid))
}
SVM_TEST <- function(data_train,data_valid, C=c(1),nIter=10,iPd=100){
  trainE<-c()
  validE<-c()
  for (i in 1:length(C)){
    t <- Sys.time();
    tmp <- my_nn(data_train,data_valid,C=C[i],nIter=nIter,iPd=iPd)
    trainE <- c(trainE, 1 - tmp[1])  
    validE <- c(validE, 1 - tmp[2]) 
    Sys.time() - t
    print(paste(format(Sys.time(), "%d-%m %X")," round", i, "of", length(C),"  errorT=",round(1 - tmp[1],2),"  errorV=", round(1 - tmp[2],2)) )
  }
  return( list( my_lineplot(list(trainE,validE),c("train","valid"),xlabel = "C",ylabel = "Error"),
                list(trainE,validE) ) )  
}
