### PRE-PROCESSIG
## workspace
#setwd("")

# Librerie necessarie 
library(corrplot)
library(FactoMineR)
library(factoextra)
library(rpart)
library(caret)
library(pROC)
library(rattle)
library("ROCR")

### ANALISI PRELIMINARI

#importo il dataset
nasa = read.csv("nasa.csv")

# guardo il dataset
str(nasa)

# controllo il numero di na presenti
sum(is.na(nasa))

#trasformo il target in factor
nasa$Hazardous = as.factor(nasa$Hazardous)

# controllo la distribuzione percentuale
prop.table(table(nasa$Hazardous) )

# la plotto
barplot(table(nasa$Hazardous), ylim=c(0,2000), col = c("red","blue"), 
        main="Distribuzione Target")


### CORRELAZIONE 

# trasformo il target in numeric per computare la correlazione
nasa$Hazardous = as.numeric(nasa$Hazardous)

# faccio la correlazione e relativo grafico
corr <- cor(nasa)
corrplot(corr, method = "circle", type = "lower", tl.col = "black", tl.cex =0.6)



### PCA

# tolgo il target per eseguire la PCA
nasa_pca <- nasa[,1:25]

# eseguo l'effettiva PCA

res.pca <- PCA(nasa_pca, scale.unit = TRUE, graph = TRUE)

# visualizzo gli autovalori
eig.val <- get_eigenvalue(res.pca) 
View(eig.val)

# grafico degli autovalori
fviz_eig(res.pca, addlabels = TRUE,  title="Percentuale di varianza per ogni dimensione")

#Visualizzo il grafo della PCA in base a variabili
var <- get_pca_var(res.pca) 
fviz_pca_var(res.pca, select.var = list(cos2=15), repel=TRUE)

