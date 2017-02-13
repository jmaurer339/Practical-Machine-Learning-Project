##
#Initial Script For Practical Machine Learning Course Project
##


#Citation
# Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity 
# Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference
# in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
##  http://groupware.les.inf.puc-rio.br/har.                                                                                                                                                                                                  Read more: http://groupware.les.inf.puc-rio.br/har#ixzz4YV10HqZ3
                                                                                                                                                                                                     
options( stringsAsFactors=F )                #set global option to import numeric data
library(caret)
set.seed(2233)                               #set seed for reproducibility

alltrain = read.csv("pml-training.csv")      # Reads Training data from Working Directory
fintest  = read.csv("pml-testing.csv")       # Reads Final Test data from Working Directory

#Further Split Training Data Set Provided Into Training and Testing for Model Development
inTrain  = createDataPartition(alltrain$classe, p = 3/4)[[1]]
training = alltrain[ inTrain,]
testing  = alltrain[-inTrain,]


#Exploratory Data Analysis on Training Data
table(training$classe)

# "Class A corresponds to the specified execution of the exercise, while the other 4 
# classes correspond to common mistakes." 

#Data Cleaning 
#Examine structure of data set
str(training)

#Cleaning: Eliminate Low Variance Predictors 
trainingNZV <- nearZeroVar(training)
training    <- training[ , -trainingNZV]

##Cleaning: Eliminate Predictors With 10% or more NA
training <- training[ lapply( training, function(x) sum(is.na(x)) / length(x) ) < 0.1 ]

##Cleaning: Eliminate Some Specific Fields In Training data
training <- training[-c(1, 2, 3, 4, 5)]

##Cleaning: Factorize Classe Variable
training$classe <- as.factor(training$classe)


#Reduce testing dataset to same columns as training / similariy Factorize classe
keepcol <- colnames(training)
testing <- testing[keepcol]
testing$classe <- as.factor(testing$classe)

#Reduce fintest dataset to same columns as training
keepcol <- keepcol[ -54]           # since there is no classe field in the fintest df
fintest <- fintest[keepcol]

#Further Split Training Data Set Provided Into training a and training b
inTrain  = createDataPartition(training$classe, p = 1/2)[[1]]
traina  = training[ inTrain,]
trainb  = training[-inTrain,]

inTrain  = createDataPartition(traina$classe, p = 1/2)[[1]]
traina1  = traina[ inTrain,]
traina2  = traina[-inTrain,]

inTrain  = createDataPartition(trainb$classe, p = 1/2)[[1]]
trainb1  = trainb[ inTrain,]
trainb2  = trainb[-inTrain,]


# Begin Model Development

#Build Decicion Tree
mod_tree  <- train(classe ~ ., data = traina1, method = "rpart")
confusionMatrix(mod_tree)

#Build Random Forest
mod_rf  <- train(classe ~ ., data = traina1, method = "rf")
confusionMatrix(mod_rf)

#Predict against one of the other training sets
pred_rf <- predict(mod_rf, traina2)
confusionMatrix(pred_rf, traina2$classe)

#Predict against testing dataset
pred_rf <- predict(mod_rf, testing)
confusionMatrix(pred_rf, testing$classe)



#Predict final 20 test cases
pred_testcases <- predict(mod_rf, fintest)






