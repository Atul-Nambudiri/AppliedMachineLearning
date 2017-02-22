library('caret')
pima_data<-read.csv('pima-indians-diabetes.data.txt')
pima_x<-pima_data[,-c(9)] #Split data into Features(pima_x) and Classification(pima_y)
pima_y<-pima_data[,9]

for (i in c(3, 5, 6, 8)) {      #Set all values in columns 3,5,6, and 8 that are 0 to be NA. These are ignored in future calculations
    zero_element<-pima_x[, i]==0
    pima_x[zero_element, i]=NA
}

cross_count<-10
training_accuracy<-array(dim=cross_count)
testing_accuracy<-array(dim=cross_count)

for (i in 1:cross_count) {      #Do 10 splits
    partition<-createDataPartition(pima_data[,9], p=.8, list=FALSE) #Split into training/testing(80/20) splits
    data<-pima_x

    training_x<-data[partition,]    #Section the data into the respective splits
    training_y<-pima_y[partition]
    testing_x<-data[-partition,]
    testing_y<-pima_y[-partition]

    pos_x<-training_x[training_y>0,]    #Get Positive and 0 sections for training features
    neg_x<-training_x[training_y==0,]
    pos_prob<-length(pos_x[,1])/length(data[,1])    #Get P(+) and P(-) for the training section
    neg_prob<-length(neg_x[,1])/length(data[,1])

    pos_mean<-sapply(pos_x, mean, na.rm=TRUE)   #Get positive and negative mean and sds
    neg_mean<-sapply(neg_x, mean, na.rm=TRUE)
    pos_sd<-sapply(pos_x, sd, na.rm=TRUE)
    neg_sd<-sapply(neg_x, sd, na.rm=TRUE)

    #Normal distribution calculations for the positive and negative training sets
    pos_training_first_terms<-t((t(training_x)-pos_mean)/pos_sd)    #Calculate (x-mean)/sd for the positive training terms
    pos_training_prob<--(1/2)*rowSums(apply(pos_training_first_terms, 1:2, function(x)x^2), na.rm=TRUE)-sum(log(sqrt(2*pi)*pos_sd)) + log(pos_prob)

    neg_training_first_terms<-t((t(training_x)-neg_mean)/neg_sd) #Calculate (x-mean)/sd for the negative training terms
    neg_training_prob<--(1/2)*rowSums(apply(neg_training_first_terms, 1:2, function(x)x^2), na.rm=TRUE)-sum(log(sqrt(2*pi)*neg_sd)) + log(neg_prob)

    positive_training<-pos_training_prob>neg_training_prob  #Find the rows which are most likely categorized as 1
    correct_training<-positive_training==training_y #Find the rows which were correctly categorized for the training examples
    training_accuracy[i] <-sum(correct_training)/length(correct_training) #Calculate training accuracy

    #Normal distribution calculations for the positive and negative testing sets
    pos_testing_first_terms<-t((t(testing_x)-pos_mean)/pos_sd) #Calculate (x-mean)/sd for the positive testing terms
    pos_testing_prob<--(1/2)*rowSums(apply(pos_testing_first_terms, 1:2, function(x)x^2), na.rm=TRUE)-sum(log(sqrt(2*pi)*pos_sd)) + log(pos_prob)

    neg_testing_first_terms<-t((t(testing_x)-neg_mean)/neg_sd) #Calculate (x-mean)/sd for the negative testing terms
    neg_testing_prob<--(1/2)*rowSums(apply(neg_testing_first_terms, 1:2, function(x)x^2), na.rm=TRUE)-sum(log(sqrt(2*pi)*neg_sd)) + log(neg_prob)
   
    positive_testing<-pos_testing_prob>neg_testing_prob #Find the rows which are most likely categorized as 1
    correct_testing<-positive_testing==testing_y #Find the rows which were correctly categorized for the testing examples
    testing_accuracy[i]<-sum(correct_testing)/length(correct_testing) #Calculate testing accuracy
}

print(training_accuracy)
print(mean(training_accuracy))
print(sd(training_accuracy))

print(testing_accuracy)
print(mean(testing_accuracy))
print(sd(testing_accuracy))