library('klaR')
library('caret')

cleveland_data<-read.csv('processed.cleveland.data.txt') #I have removed all rows that have a ? for any attribute
cleveland_x<-cleveland_data[,-c(14)] #Split data into Features(cleveland_x) and Classification(cleveland_y)
cleveland_y<-cleveland_data[,14]

cleveland_y<-as.factor(cleveland_y)

cross_count<-10
accuracy<-array(dim=cross_count)

for (i in 1:cross_count) {
    partition<-createDataPartition(cleveland_data[,14], p=.85, list=FALSE) #Split the data into a training/testing(85/15) split

    training_x<-cleveland_x[partition,] #Section the data into the respective splits
    training_y<-cleveland_y[partition]
    testing_x<-cleveland_x[-partition,]
    testing_y<-cleveland_y[-partition]

    #Train a naives bayes with 10-fold k-fold cross_validation
    bayes_model<-train(training_x, training_y, 'nb', trControl=trainControl(method='cv', number=10))

    predictions<-predict(bayes_model,newdata=testing_x)  #Test the model on the testing set
    
    correct_testing<-predictions==testing_y #Find the rows which were correctly categorized for the testing examples
    accuracy[i]<-sum(correct_testing)/length(correct_testing) #Calculate the accuracy for the split

    print(confusionMatrix(predictions, testing_y))  #Print out the confusion matrix to see where the confusion is
}

print(accuracy)
print(mean(accuracy))
print(sd(accuracy))


