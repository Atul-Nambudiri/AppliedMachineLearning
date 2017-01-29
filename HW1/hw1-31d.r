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

#Train an svm_model on the training data
svm_model<-svmlight(training_x, training_y)

#Test out the trained model on the testing set
predict<-predict(svm_model, newdata=testing_x)
predictions<-predict$class
confusionMatrix(data=predictions, testing_y)


