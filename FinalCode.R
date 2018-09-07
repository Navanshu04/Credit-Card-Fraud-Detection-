
rm(list=ls()) ; gc()

library(ggplot2) 
library(readr) 
library(caret)
library(e1071)
library(randomForest)
library(ROSE)
library(rpart)
library(rpart.plot)
library(ROCR)
library(caTools)
library(magrittr)     # Data pipelines: %>% %T>% %<>%.
library(unbalanced)   # Resampling using ubSMOTE.
library(rattle)       # Draw fancyRpartPlot().
library(nnet)         # Model: neural network.
library(stringi)      # For %s+% (string concatenate)
library(readr)        # for readr
library(dplyr)        # for sample_n
library(GGally)
library(Boruta)	# Wrapper around randomForest, discovers important_features
library(Metrics)      # For rmse()
library(forcats)


card <- read_csv("C:/Users/DELL 5548/Desktop/Major Project/creditcard.csv")
str(card)
summary(card)
# convert class variable to factor

card$Class <- factor(card$Class)

set.seed(1)
split <- sample.split(card$Class, SplitRatio = 0.7)

train <- subset(card, split == T)
cv <- subset(card, split == F)

table(cv$Class)

##downsampling And oversampling the positive data samples to avoid data imbalance

data_balanced_under <- ovun.sample(Class ~ ., data = train, method = "under", N = 688, seed = 1)$data
table(data_balanced_under$Class)

data_balanced_over <- ovun.sample(Class ~ ., data = train, method = "over", N = 398038, seed = 1)$data
table(data_balanced_over$Class)


data_balanced_both <- ovun.sample(Class ~ ., data = train, method = "both", N = 85443, seed = 1)$data
table(data_balanced_both$Class)

## logistic regression

glm.model <- glm(Class ~ ., data =data_balanced_both, family = "binomial")
glm.predict <- predict(glm.model, cv, type = "response")
table(cv$Class, glm.predict > 0.5)
cm_lr<-confusionMatrix(glm.predict,cv$Class, positive = "0")
acc_lr <- cm_lr$overall['Accuracy']
ROCRpred <- prediction(glm.predict, cv$Class)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf,colorize=T,main="ROC Curve")
abline(a=0,b=1)

cm_lg<-confusionMatrix(data = as.numeric(glm.predict>0.5), reference = cv$Class)
acc_lg <- cm_lg$overall['Accuracy']

## Decision tree model

tree.model <- rpart(Class ~ ., data = data_balanced_under, method = "class", minbucket = 20)
prp(tree.model) 
tree.predict <- predict(tree.model, cv, type = "class")
table(cv$Class, tree.predict)
tree<-confusionMatrix(tree.predict,cv$Class, positive = "0")
acc_tree <- tree$overall['Accuracy']

#SVM Model 

model_svm <- train(Class~.,data=data_balanced_under,method="svmRadial",trControl=trainControl(method='cv'))
pred_svm <- predict(model_svm, cv)
table(cv$Class, pred_svm)
cm_svm <- confusionMatrix(pred_svm,cv$Class, positive = "0")
cm_svm
acc_svm <- cm_svm$overall['Accuracy']
plot(pred_svm,data=data_balanced_under)
#Random Forest

rf=randomForest(Class~.,data=data_balanced_under)
pred=predict(rf,cv)
table(cv$Class,pred)
cm_rf<-confusionMatrix(pred,cv$Class, positive = "0")
acc_rf <- cm_rf$overall['Accuracy']
plot(rf)
attributes(rf)
hist(treesize(rf),main="Number of Nodes for the Tree",col="green")

#Variables Playing Important Role in Accuracy

varImpPlot(rf,sort = T,n.var = 10,main="Important Variables")
importance(rf)
varUsed(rf)
qplot(Time,Amount,data=card,color=Class)

acc_list = c(acc_svm,acc_rf,acc_tree,acc_lg)
acc_list_names <- c("SVM","RandomForest","Decison Tree","Logistic Regression")

## Plot the Accuracy from different Models
colors <- c("darkolivegreen4","orange","brown","cadetblue1")
barplot(acc_list,names.arg = acc_list_names,col = colors,ylim=c(0,1.1),xlab = "Model",ylab="Accuracy")


##Flling empty values

sum(is.na(card)) # to find out null/empty values

mynumcols<-card[,-1] %>%
  sapply(is.numeric) %>%      # Output is TRUE, FALSE...
  which() %>%                 # Output is col index wherever input is TRUE
  names()                     # Outputs col names for an input index
mynumcols

# Find if there are any missing values and do a rough fix as nnet will not work
# na.roughhfix() only fixes numerical values by mean/median
card[,mynumcols] %<>% na.roughfix() 

# Initialise random numbers for repeatable results.
set.seed(123)

##Feature Plot

#Partition the full dataset into two. Stratified sampling.
mytraining<-createDataPartition(card$Class, p=0.7,list=FALSE)
myvalid<- card[-mytraining,]


#Separate predictors and target in training data
XX<-card[,-c(1,31)]    
YY<-as.factor(card$Class) # Class col is target

#Balance training dataset now using ubSMOTE 
bal_ccdata <- ubSMOTE(X = XX, Y = YY,   # Also y be a vector not a dataframe
                      perc.over=200,   #200,
                      perc.under=800,  #800,
                      k=3,
                      verbose=TRUE) 
#7.1 ubSMOTE returns balanced data frame which is list-structure. so doing a cbind 
mytraindata <- cbind(bal_ccdata$X, Class = bal_ccdata$Y)


#Check the dropping-out proportion
table(card$Class)/nrow(card)               # Earlier

table(mytraindata$Class)/nrow(mytraindata)     # Now

#As the dataset is large, reducing data to 1000 records
myreduced_ccdata<-mytraindata %>% sample_n(1000)
save(myreduced_ccdata, file= "myreduced_ccdata")

for (i in seq(from =1, to = 28, by = 2))
{
  show(
    featurePlot(
      x = myreduced_ccdata[, c(i,i+1)], 
      y = myreduced_ccdata$Class,
      plot = "density", 
      scales = list(x = list(relation="free"), y = list(relation="free")), 
      adjust = 1.5, # Adjusts curve smoothness
      pch =  c(1, 8, 15),    # Point character at the bottom of graph to show pt density
      layout = c(2,1 ),   # Four columns
      auto.key=TRUE
    )
  )
}

for (i in seq(from =1, to = 28, by = 2))
{
  show(
    featurePlot(
      x = myreduced_ccdata[, c(i,i+1)], 
      y = myreduced_ccdata$Class,
      plot = "box", jitter = TRUE,
      col=c("Orange"),
      scales = list(x = list(relation="free"), 
                    y = list(relation="free")), 
      adjust = 1.5, # Adjusts curve smoothness
      aspect = "fill",
      
      pch =  c( 8, 19, 17),    # Point character at the bottom of graph to show pt density
      layout = c(4,1 ),   # Four columns
      auto.key=TRUE
    )
  )
}

# Variable Importance Plot
varImpPlot(rf, sort = T, main = "Variable Importance Plot", n.var = 5)

# Variable Importance Table
myvarImptable <- data.frame(importance(rf, type=2))
myvarImptable$variables <- row.names(myvarImptable)
myvarImptable[order(myvarImptable$MeanDecreaseGini, decreasing = T),]

