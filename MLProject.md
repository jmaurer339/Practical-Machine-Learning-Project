<center>
#Practical Machine Learning Project      
###*Jim Maurer*</br>*February 2017*
</center>
</br>
**Summary** </br>
Human Activity Recognition (HAR) is an increasingly important area of research. In this research devices are attached to the human body. While the body is in motion these devices record an array of data. Commercially available products such as Jawbone Up, Nike FuelBand, and Fitbit are examples of such devices. For more extensive research more complex devices are used to capture a much wider variety of data. Much of the data collected using these devices is used simply to track performance, that is to quantify **how much** activity is completed. The data analyzed below was collected with the objective of understanding the quality, that is to quantify **how well** the activity is completed. 

The data used in this study were collected from positioning four such devices called accelerometers in the waist, left thigh, right ankle, and right arm of six test subjects. Subjects were then asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). Class A corresponds to the correct execution of the exercise, while the other 4 classes correspond to common mistakes.

The following analysis examines whether or not it is possible to utilize HAR data to predict, based on measures obtained from the accelerometers (position, pitch, roll, skew, etc.) the quality of the exercise; that is which of the five different fashions or classes were executed.

**Citation** </br>
Data, meta information and background on this study were obtained from the following source: Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. *Qualitative Activity 
Recognition of Weight Lifting Exercises*. Proceedings of 4th International Conference
in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
(http://groupware.les.inf.puc-rio.br/har).
</br>
</br>
**Set global option to import numeric data, load caret package, and set seed for reproducibility**

```r
options( stringsAsFactors=F )     
library(caret)
set.seed(2233)  
```

**Read in data**

```r
alltrain = read.csv("pml-training.csv")     
fintest  = read.csv("pml-testing.csv")
```

**Futher split Training data into training and testing data**

```r
inTrain  = createDataPartition(alltrain$classe, p = 3/4)[[1]]
training = alltrain[ inTrain,]
testing  = alltrain[-inTrain,]
```

##Exploratory Analysis
**Examine structure of data set (output surpressed)**

```r
str(training)
```

**Report dimensionality of training data and one or two interesting variables**

```r
dim(training)
```

```
## [1] 14718   160
```

```r
table(training$classe)
```

```
## 
##    A    B    C    D    E 
## 4185 2848 2567 2412 2706
```

```r
table(training$user_name)
```

```
## 
##   adelmo carlitos  charles   eurico   jeremy    pedro 
##     2921     2372     2639     2308     2532     1946
```

##Data Cleaning
**Eliminate Low Variance Predictors** 

```r
trainingNZV <- nearZeroVar(training)
training    <- training[ , -trainingNZV]
```

**Eliminate Predictors With 10% or more NA**

```r
training <- training[ lapply( training, function(x) sum(is.na(x)) / length(x) ) < 0.1 ]
```

**Eliminate Some Specific Fields In Training data - by definition not useful for prediction**   

```r
training <- training[-c(1, 2, 3, 4, 5)]
```

**Factorize Classe Variable**

```r
training$classe <- as.factor(training$classe)
```



##Apply Similar Transformations To Testing data And to fintest (case) data  
**Reduce dimensions of testing dataset to same as training / similariy Factorize classe**  

```r
keepcol <- colnames(training)
testing <- testing[keepcol]
testing$classe <- as.factor(testing$classe)
```


**Perform similar dimension reduction on fintest (final 20 cases) dataset**

```r
keepcol <- keepcol[ -54]          
fintest <- fintest[keepcol]
```

**Re-examine dimensionality of training data**

```r
dim(training)
```

```
## [1] 14718    54
```

**Further split Training data**  </br>
Since the Training data still has over 14,000 rows and 53 predictors, it is too large to reasonably process alternative algorithms (on my local machine). So the Training data is further split into four separate training data sets.

```r
inTrain  = createDataPartition(training$classe, p = 1/2)[[1]]
traina  = training[ inTrain,]
trainb  = training[-inTrain,]

inTrain  = createDataPartition(traina$classe, p = 1/2)[[1]]
traina1  = traina[ inTrain,]
traina2  = traina[-inTrain,]

inTrain  = createDataPartition(trainb$classe, p = 1/2)[[1]]
trainb1  = trainb[ inTrain,]
trainb2  = trainb[-inTrain,]
```

#Begin Model Development
**Fit a single decision tree to the traina1 data**

```r
mod_tree  <- train(classe ~ ., data = traina1, method = "rpart")
```

```
## Loading required package: rpart
```

```r
confusionMatrix(mod_tree)
```

```
## Bootstrapped (25 reps) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 23.3  3.5  1.2  2.1  1.2
##          B  1.3  8.7  1.1  2.3  2.6
##          C  2.4  4.8 13.2  6.1  3.4
##          D  1.1  1.9  1.7  5.5  1.2
##          E  0.3  0.4  0.2  0.5 10.0
##                             
##  Accuracy (average) : 0.6073
```
The within sample accuracy of the tree can be seen to be about 60%.
</br>

**Fit A Random Forest**

```r
mod_rf  <- train(classe ~ ., data = traina1, method = "rf")
confusionMatrix(mod_rf)
```

```
## Bootstrapped (25 reps) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.2  0.8  0.0  0.0  0.0
##          B  0.0 18.2  0.6  0.1  0.2
##          C  0.0  0.3 16.8  0.4  0.1
##          D  0.0  0.1  0.1 15.8  0.3
##          E  0.0  0.0  0.0  0.1 17.9
##                             
##  Accuracy (average) : 0.9695
```
The within sample accuracy of the random forest can be seen to be about 97%, a significant improvement over the single decision tree's predictive ability. 
</br>


##Out of Sample Validation
**Predict against one of the other training sets**

```r
pred_rf <- predict(mod_rf, traina2)
confusionMatrix(pred_rf, traina2$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1046    7    0    0    1
##          B    0  691   10    1    4
##          C    0   13  631   19    2
##          D    0    1    1  582    5
##          E    0    0    0    1  664
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9823          
##                  95% CI : (0.9775, 0.9863)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9776          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9705   0.9829   0.9652   0.9822
## Specificity            0.9970   0.9949   0.9888   0.9977   0.9997
## Pos Pred Value         0.9924   0.9788   0.9489   0.9881   0.9985
## Neg Pred Value         1.0000   0.9929   0.9964   0.9932   0.9960
## Prevalence             0.2843   0.1935   0.1745   0.1639   0.1837
## Detection Rate         0.2843   0.1878   0.1715   0.1582   0.1805
## Detection Prevalence   0.2865   0.1919   0.1808   0.1601   0.1808
## Balanced Accuracy      0.9985   0.9827   0.9858   0.9814   0.9910
```
The out-of-sample sample accuracy of the random forest as measured against traina2 is 98%. The ability of this model to predict outside of the sample of data used to build the algorithmic is confirmed. The traina2 dataset had just over 3679 cases. However, we can confirm this by applying the model to the entire testing dataset with 4904 cases.     
</br>

**Predict against testing dataset**

```r
pred_rf <- predict(mod_rf, testing)
confusionMatrix(pred_rf, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395   16    0    0    2
##          B    0  914   19    3    2
##          C    0   18  833   13    3
##          D    0    1    3  788    4
##          E    0    0    0    0  890
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9829          
##                  95% CI : (0.9788, 0.9863)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9783          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9631   0.9743   0.9801   0.9878
## Specificity            0.9949   0.9939   0.9916   0.9980   1.0000
## Pos Pred Value         0.9873   0.9744   0.9608   0.9899   1.0000
## Neg Pred Value         1.0000   0.9912   0.9946   0.9961   0.9973
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1864   0.1699   0.1607   0.1815
## Detection Prevalence   0.2881   0.1913   0.1768   0.1623   0.1815
## Balanced Accuracy      0.9974   0.9785   0.9829   0.9891   0.9939
```
The initial results of the out-of-sample validation as seen with the traina2 dataset have been confirmed against the 4904 cases in the testing dataset. The model had an accuracy of 98% in this sample as well. 
</br>
  
##Conclusion 

Using a random forest it has been shown that it is possible to use data obtained from 
accelerometers to accurately predict the manner in which an exercise was completed. An
algorithm using this approach could be applied to provide a person with real-time feedback
on the quality with which they are executing a particular routine. 
</br>
</br>
</br>
