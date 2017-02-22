library('klaR')
library('caret')

k9_data<-read.csv('K9.data')

k9_data <- k9_data[,1:5409]
missing_rows <- apply(k9_data, 1, function(x) all( x != '?'))
k9_data <- k9_data[missing_rows,]

k9_data <- sapply( k9_data, as.numeric ) # Change inactive and active to 1 and 2
k9_data[,5409][k9_data[,5409]==2] <- -1  # Change all the 2's to -1's

k9_x<-k9_data[,-c(5409)] #Split data into Features(k9_x) and Classification(k9_y)
k9_y<-as.factor(k9_data[,5409])

partition1<-createDataPartition(k9_data[,5409], p=.5, list=FALSE) #Split the data into a training/testing(50/50) split

training_x<-k9_x[partition1,] #Section the data into the respective splits
training_y<-k9_y[partition1]
testing_x<-k9_x[-partition1,]
testing_y<-k9_y[-partition1]

#Train a naives bayes with 10-fold k-fold cross_validation
bayes_model<-train(training_x, training_y, 'nb', trControl=trainControl(method='cv', number=10))
print(bayes_model) #Print out the predicted accuracy from the cross-validation

predictions<-predict(bayes_model,newdata=testing_x) #Test the model on the testing set
confusionMatrix(data=predictions, testing_y)