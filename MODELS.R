# Librerie
library(caret)
library(nnet)
library(ROCR)
library(e1071)

# workspace
#setwd("")
nasa_num <- read.csv("nasa.csv", stringsAsFactors = TRUE)

#scelgo le covariate in base alla PCA 
nasa.reduced <- subset(nasa_num, select = c(2,4,10,12,14,15,19,25))

### --- SPLIT DEI DATI ---
# funzione necessaria allo split dei dati
split.data <- function(data, p = 0.7, s = 1){
  set.seed(s)
  index = sample(1:dim(data)[1])
  train = data[index[1:floor(dim(data)[1] * p)], ]
  test = data[index[((ceiling(dim(data)[1] * p)) + 1):dim(data)[1]], ] 
  return(list(train=train, test=test)) }

# Effettuo lo split dei dati utilizzando lo stesso seed degli altri modelli
set.seed(7) 
allset<- split.data(nasa.reduced, p=0.7, s=7)
trainset <- allset$train
testset <- allset$test

# conto le occorrenze del target in train e test
table(trainset$Hazardous)
table(testset$Hazardous)

# percentuale di occorrenze, simile 
prop.table(table(trainset$Hazardous)) 
prop.table(table(testset$Hazardous)) 


### --- 10-FOLD CROSS-VALIDATION ---

# Preparo il controllo per la 10-fold
control = trainControl(method = "repeatedcv", number = 10,repeats = 10, classProbs = TRUE, summaryFunction = twoClassSummary)


### ---  NAIVE BAYES ---

## train del modello Naive Bayes
nb.model <- train(Hazardous~., data=trainset, trControl=control, method="naive_bayes", metric = "ROC",prob=TRUE)

# vector and dataframe of prediction
nb.model.raw = predict(nb.model, testset[,! names(testset) %in% c("Hazardous")], type="raw",probability=TRUE)
nb.model.prob = predict(nb.model, testset[,! names(testset) %in% c("Hazardous")], type="prob",probability=TRUE)

# confusion matrix and accuracy
nb.model.cM.F = confusionMatrix(table(nb.model.raw, testset$Hazardous), mode = "prec_recall", positive="False")
nb.model.cM.F.acc = nb.model.cM.F$overall['Accuracy']
print(paste0("Accuracy for Naive Bayes in test set: ",nb.model.cM.F.acc))

## ROC Naive Bayes
pred.rocr.nb.model =  ROCR::prediction(nb.model.prob[,2], testset$Hazardous)
perf.rocr.nb.model = performance(pred.rocr.nb.model, measure = "auc", x.measure = "cutoff") 
perf.tpr.rocr.nb.model = performance(pred.rocr.nb.model, "tpr","fpr")
# disegnamo la curva roc
png(file = "img/Curva_AUC_NB.png", width = 800, height = 600, units = "px")
plot(perf.tpr.rocr.nb.model, colorize=T,main=c("Naive Bayes",paste("AUC:",(perf.rocr.nb.model@y.values))))
abline(a=0, b=1)
garbage <- dev.off()



### --- NEURAL NETWORK --- 

# train
NN.model <- train(Hazardous~., data=trainset, trControl=control, method="nnet", metric = "ROC", type="Classification")

# vector and dataframe of prediction
NN.model.raw = predict(NN.model, testset[,! names(testset) %in% c("Hazardous")], type="raw",probability=TRUE)
NN.model.prob = predict(NN.model, testset[,! names(testset) %in% c("Hazardous")], type="prob",probability=TRUE)

# confusion matrix and accuracy
NN.model.cM.F = confusionMatrix(table(NN.model.raw, testset$Hazardous), mode = "prec_recall", positive="False")
NN.model.cM.F.acc = NN.model.cM.F$overall['Accuracy']
print(paste0("Accuracy for Neural Network in test set: ",NN.model.cM.F.acc),)

## ROC Neural Network
pred.rocr.NN.model =  ROCR::prediction(NN.model.prob[,2], testset$Hazardous)
perf.rocr.NN.model = performance(pred.rocr.NN.model, measure = "auc", x.measure = "cutoff") 
perf.tpr.rocr.NN.model = performance(pred.rocr.NN.model, "tpr","fpr")
print(c("AUC Neural Network :",perf.rocr.NN.model@y.values[[1]]))

# tuning 
nnetGrid <-  expand.grid(size = seq(from = 1, to = 7, by = 1),decay = seq(from = 0.1, to = 0.5, by = 0.1))
NN.model2 <- train(Hazardous~., data=trainset, trControl=control, method="nnet", metric = "ROC", type="Classification", tuneGrid=nnetGrid)

# vector and dataframe of prediction
NN.model2.raw = predict(NN.model2, testset[,! names(testset) %in% c("Hazardous")], type="raw",probability=TRUE)
NN.model2.prob = predict(NN.model2, testset[,! names(testset) %in% c("Hazardous")], type="prob",probability=TRUE)

# confusion matrix and accuracy
NN.model2.cM.F = confusionMatrix(table(NN.model2.raw, testset$Hazardous), mode = "prec_recall", positive="False")
NN.model2.cM.F.acc = NN.model2.cM.F$overall['Accuracy']
print(paste0("Accuracy for Neural Network 2 in test set: ",NN.model2.cM.F.acc))

## ROC Neural Network
pred.rocr.NN.model =  ROCR::prediction(NN.model2.prob[,2], testset$Hazardous)
perf.rocr.NN.model = performance(pred.rocr.NN.model, measure = "auc", x.measure = "cutoff") 
perf.tpr.rocr.NN.model = performance(pred.rocr.NN.model, "tpr","fpr")
print(c("AUC Neural Network 2:",perf.rocr.NN.model@y.values[[1]]))

# disegnamo la curva roc
png(file = "img/Curva_AUC_NN.png",width = 800, height = 600, units = "px")
plot(perf.tpr.rocr.NN.model, colorize=T,main=c("Neural Network", paste("AUC:",(perf.rocr.NN.model@y.values))))
abline(a=0, b=1)
garbage <- dev.off()



### --- SVM ---

### --- SVM RADIAL --- 
# train
svm.model.rad <- train(Hazardous~., data=trainset, trControl=control, method="svmRadial", metric = "ROC")

# vector and dataframe of prediction
svm.model.rad.raw = predict(svm.model.rad, testset[,! names(testset) %in% c("Hazardous")], type="raw")
svm.model.rad.prob = predict(svm.model.rad, testset[,! names(testset) %in% c("Hazardous")], type="prob")

# confusion matrix and accuracy)
svm.model.rad.cM.F = confusionMatrix(table(svm.model.rad.raw, testset$Hazardous), mode = "prec_recall", positive="False")
svm.model.rad.cM.F.acc = svm.model.rad.cM.F$overall['Accuracy']
print(paste0("Accuracy for svm radial in test set: ",svm.model.rad.cM.F.acc))



### --- SVM POLYNOMIAL --- 
# train
#set.seed(7)
#
#svm.model.pol <- train(Hazardous~., data=trainset, trControl=control, method="svmPoly", metric = "ROC", cost=1)
#
svm.model.pol <- train(Hazardous~., data=trainset, trControl=control, method="svmPoly", metric = "ROC")

# vector and dataframe of prediction
svm.model.pol.raw = predict(svm.model.pol, testset[,! names(testset) %in% c("Hazardous")], type="raw")
svm.model.pol.prob = predict(svm.model.pol, testset[,! names(testset) %in% c("Hazardous")], type="prob")

# confusion matrix and accuracy
svm.model.pol.cM.F = confusionMatrix(table(svm.model.pol.raw, testset$Hazardous), mode = "prec_recall", positive="False")
svm.model.pol.cM.F.acc = svm.model.pol.cM.F$overall['Accuracy']
print(paste0("Accuracy for svm polynimial in test set: ",svm.model.pol.cM.F.acc))



### --- SVM Linear --- 
# train
svm.model.lin <- train(Hazardous~., data=trainset, trControl=control, method="svmLinear", metric = "ROC")

# vector and dataframe of prediction
svm.model.lin.raw = predict(svm.model.lin, testset[,! names(testset) %in% c("Hazardous")], type="raw")
svm.model.lin.prob = predict(svm.model.lin, testset[,! names(testset) %in% c("Hazardous")], type="prob")

# confusion matrix and accuracy
svm.model.lin.cM.F = confusionMatrix(table(svm.model.lin.raw, testset$Hazardous), mode = "prec_recall", positive="False")
svm.model.lin.cM.F.acc = svm.model.lin.cM.F$overall['Accuracy']
print(paste0("Accuracy for svm linear in test set: ",svm.model.lin.cM.F.acc))


### ---  TUTTI I VALORI DI ACCURATEZZA delle SVM
print(c(paste0("Accuracy for SVM radial in test set:     ",svm.model.rad.cM.F.acc), 
        paste0("Accuracy for SVM polinomial in test set: ",svm.model.pol.cM.F.acc),
        paste0("Accuracy for SVM radial in test set:     ",svm.model.lin.cM.F.acc)))


## ROC Svm Radial
pred.rocr.svm.model.rad =  ROCR::prediction(svm.model.rad.prob[,2], testset$Hazardous)
perf.rocr.svm.model.rad = performance(pred.rocr.svm.model.rad, measure = "auc", x.measure = "cutoff") 
perf.tpr.rocr.svm.model.rad = performance(pred.rocr.svm.model.rad, "tpr","fpr")
# disegnamo la curva roc
png(file = "img/Curva_AUC_SVM_RAD.png",width = 800, height = 600, units = "px")
plot(perf.tpr.rocr.svm.model.rad, colorize=T,main=c("Svm Radial",paste("AUC:",(perf.rocr.svm.model.rad@y.values))))
abline(a=0, b=1)
garbage <- dev.off()

## ROC Svm Polynomial
pred.rocr.svm.model.pol =  ROCR::prediction(svm.model.pol.prob[,2], testset$Hazardous)
perf.rocr.svm.model.pol = performance(pred.rocr.svm.model.pol, measure = "auc", x.measure = "cutoff") 
perf.tpr.rocr.svm.model.pol = performance(pred.rocr.svm.model.pol, "tpr","fpr")
# disegnamo la curva roc
png(file = "img/Curva_AUC_SVM_POL.png",width = 800, height = 600, units = "px")
plot(perf.tpr.rocr.svm.model.pol, colorize=T,main=c("Svm Polynomial",paste("AUC:",(perf.rocr.svm.model.pol@y.values))))
abline(a=0, b=1)
garbage <- dev.off()

## ROC Svm Linear
pred.rocr.svm.model.lin =  ROCR::prediction(svm.model.lin.prob[,2], testset$Hazardous)
perf.rocr.svm.model.lin = performance(pred.rocr.svm.model.lin, measure = "auc", x.measure = "cutoff") 
perf.tpr.rocr.svm.model.lin = performance(pred.rocr.svm.model.lin, "tpr","fpr")
# disegnamo la curva roc
png(file = "img/Curva_AUC_SVM_LIN.png",width = 800, height = 600, units = "px")
plot(perf.tpr.rocr.svm.model.lin, colorize=T,main=c("Svm Linear",paste("AUC:",(perf.rocr.svm.model.lin@y.values))))
abline(a=0, b=1)
garbage <- dev.off()


# CONFRONTO TRA LE ROC delle SVM
png(file = "img/Curva_ROC_SVM.jpg",width = 800, height = 600, units = "px")
plot(perf.tpr.rocr.svm.model.lin, col="blue")
par(new=T)
plot(perf.tpr.rocr.svm.model.pol, col="orange")
par(new=T)
plot(perf.tpr.rocr.svm.model.rad, col="green")
abline(a=0, b=1)
legend(0.4, 0.25, legend=c("Svm Lineare","Svm Polinomiale","Svm Radiale"), col=c("blue","orange", "green"), lty=1:1, cex=0.9)
garbage <- dev.off()
# vediamo che radial è la migliore anche qui e consideriamo solo quella?


### --- CONFRONTO TRA LE CURVE ROC ---

png(file = "img/Curva_ROC_Comp.png",width = 800, height = 600, units = "px")
plot(perf.tpr.rocr.nb.model, col="red", option="c")
par(new=T)
plot(perf.tpr.rocr.svm.model.rad, col="green")
par(new=T)
plot(perf.tpr.rocr.NN.model, col="violet")
abline(a=0, b=1)
legend(0.4, 0.25, legend=c("Naive Bayes", "Svm Radiale","Neural Network"),
       col=c("red","green","violet"), lty=1:1, cex=0.9)
garbage <- dev.off()


#Confronto tempistiche dei i modelli
cv.values = resamples(list(nb=nb.model, nn = NN.model, svm= svm.model.rad)) 
summary(cv.values) 
# tempo di traing totale e parziale per modello
times = cv.values$timings[1:2]
print(times)

# vari plot
dotplot(cv.values, metric = "ROC") 
bwplot(cv.values, layout = c(3, 1))  

