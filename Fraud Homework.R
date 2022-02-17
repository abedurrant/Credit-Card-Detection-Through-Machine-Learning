#Import required library
library(MASS)
library(ranger)
library(caret)
library(pROC)


credit <- read.csv("CCFraud.csv")

credit <- na.omit(credit)
credit$Class <- as.numeric(credit$Class)

#Important metrics
#Sensitiviy(Recall) (Of the actual Fraudulent transactions, what % did we get)
# 1 - Specificity (Of the real transactions, how many did we label as fradulent)
#Postive Predictive Value (Of those we said are fradulent, how many actually were)

#Data cleaning and initial plots
sort(cor(credit[,2:30],credit[,31]))
plot(credit$V17, credit$Class)
ggplot(data = credit, mapping =aes(x=V17, y = Class))+ geom_point() + geom_smooth(se=FALSE)

#Random Forest
samp <- sample(1:284802, 56000)
test <- credit[samp,]
train <- credit[-samp,]
forest <- ranger(Class ~ .,data=credit,num.trees = 100, importance = "permutation")
forest$confusion.matrix[2,2]/(forest$confusion.matrix[1,2]+forest$confusion.matrix[2,2])
forest <- ranger(Class ~ .,data=credit,num.trees = 1000, importance = "permutation", probability = TRUE)
importance(forest)
preds_all = forest$predictions[,2]>.05
preds_all = as.numeric(preds_all)
preds_all = as.factor(preds_all)
roc(as.numeric(credit$Class), as.numeric(preds_all))
confusionMatrix(data = as.factor(preds_all), reference = credit$Class)

#SVM
library(e1071)
model = svm(Class ~ ., data=credit, kernel = "polynomial")
summary(model)
p = predict(model,newdata = credit)
auc(credit$Class, as.numeric(p)-1)
confusionMatrix(data = p, reference = as.factor(credit$Class))

#NDA
library(MASS)
model5 = qda(Class ~ ., data=credit)
p2 = predict(model5,newdata = credit)
preds_qda = p2$posterior[,2]>.999
preds_qda = as.numeric(preds_qda)
preds_qda = as.factor(preds_qda)
auc((credit$Class), as.numeric(preds_qda))
confusionMatrix(data = preds_qda, reference = as.factor(credit$Class))

#Very basic Neural Net
nn <- train(Class ~ ., train, method = "nnet", linout = F, maxit = 1000)
summary(nn)
p = predict(nn)
auc(train$Status, as.numeric(p))


#Cross Validation Random Forest
#Testing different numbers of trees
f <- c(50, 100, 500, 1000) 
#Preparing metrics
sens_plural <- rep(x=NA, times=4)
spec_inv_plural <- rep(x=NA, times=4)
ppv_plural <- rep(x=NA, times=4)
for(j in f){
  n.cv = 3
  sens <- rep(x=NA, times=n.cv)
  spec_inv <- rep(x=NA, times=n.cv)
  ppv <- rep(x=NA, times=n.cv)
  for(i in 1:n.cv) {
    #Randomly sampling parts of the data for a train/test split
    samp <- sample(1:284802, 56000)
    test <- credit[samp,]
    train <- credit[-samp,]
    forest <- ranger(Class ~ .,data=train,num.trees = j, importance = "permutation", probability = TRUE)
    preds = predict(forest, data = test)
    preds_forest = preds$predictions[,2]>.04
    preds_forest = as.numeric(preds_forest)
    preds_forest = as.factor(preds_forest)
    forest_conf = confusionMatrix(data = as.factor(preds_forest), reference = as.factor(test$Class))
    sens[i] = forest_conf$table[2,2]/(forest_conf$table[1,2]+forest_conf$table[2,2])
    spec_inv[i] = 1-(forest_conf$table[2,1]/(forest_conf$table[2,1]+forest_conf$table[1,1]))
    ppv[i] <- forest_conf$table[2,2]/(forest_conf$table[2,2]+forest_conf$table[2,1])
    
  }
  sens_plural[j] = mean(sens)
  spec_inv_plural[j] = mean(spec_inv)
  ppv_plural[j] =mean(ppv)
}

#Cross Validation QDA
n.cv = 3
sens_qda <- rep(x=NA, times=n.cv)
spec_inv_qda <- rep(x=NA, times=n.cv)
ppv_qda <- rep(x=NA, times=n.cv)
for(i in 1:n.cv) {
  samp <- sample(1:284802, 56000)
  test <- credit[samp,]
  train <- credit[-samp,]
  model = qda(Class ~ ., data=train)
  p2 = predict(model,newdata = test)
  p2$posterior[,1]
  preds_qda = p2$posterior[,2]>.9999
  preds_qda = as.numeric(preds_qda)
  preds_qda = as.factor(preds_qda)
  qda_conf = confusionMatrix(data = preds_qda, reference = as.factor(test$Class))
  sens_qda[i] = qda_conf$table[2,2]/(qda_conf$table[1,2]+qda_conf$table[2,2])
  spec_inv_qda[i] = 1-(qda_conf$table[2,1]/(qda_conf$table[2,1]+qda_conf$table[1,1]))
  ppv_qda[i] <- qda_conf$table[2,2]/(qda_conf$table[2,2]+qda_conf$table[2,1])
  
}

mean(sens_qda)
mean(spec_inv_qda)
mean(ppv_qda)


#Cross Validation SVM
n.cv = 3
sens_sv <- rep(x=NA, times=n.cv)
spec_inv_sv <- rep(x=NA, times=n.cv)
ppv_sv <- rep(x=NA, times=n.cv)
for(i in 1:n.cv) {
  samp <- sample(1:284802, 56000)
  test <- credit[samp,]
  train <- credit[-samp,]
  model = svm(Class ~ ., data=train, kernel = "polynomial")
  summary(model)
  p = predict(model,newdata = test)
  sv_conf = confusionMatrix(data = p, reference = as.factor(test$Class))
  sens_sv[i] = sv_conf$table[2,2]/(sv_conf$table[1,2]+sv_conf$table[2,2])
  spec_inv_sv[i] = 1-(sv_conf$table[2,1]/(sv_conf$table[2,1]+sv_conf$table[1,1]))
  ppv_sv[i] <- sv_conf$table[2,2]/(sv_conf$table[2,2]+sv_conf$table[2,1])
  
}

mean(sens_sv)
mean(spec_inv_sv)
mean(ppv_sv)

#Random Forest had best performances so we will use it for the predictions

#Predictions
predict_these <- read.csv("IsFraudulent.csv")
forest_final <- ranger(Class ~ .,data=credit,num.trees = 1000, importance = "permutation", probability = TRUE)
sort(importance(forest_final))
preds = predict(forest_final, data = predict_these)

#More details can be found in the report
