library('klaR')
library('caret')

pima_data<-read.csv('pima-indians-diabetes.data.txt')
pima_x<-pima_data[,-c(9)] #Split data into Features(pima_x) and Classification(pima_y)
pima_y<-as.factor(pima_data[,9])

partition<-createDataPartition(pima_data[,9], p=.8, list=FALSE) #Split the data into a training/testing(80/20) split

training_x<-pima_x[partition,] #Section the data into the respective splits
training_y<-pima_y[partition]
testing_x<-pima_x[-partition,]
testing_y<-pima_y[-partition]

#Train a naives bayes with 10-fold k-fold cross_validation
bayes_model<-train(training_x, training_y, 'nb', trControl=trainControl(method='cv', number=10))
print(bayes_model) #Print out the predicted accuracy from the cross-validation

predictions<-predict(bayes_model,newdata=testing_x) #Test the model on the testing set
confusionMatrix(data=predictions, testing_y)


