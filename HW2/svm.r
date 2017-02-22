library('caret')

k9_data<-read.csv('K9.data')

#Remove the extra column at the end, and remove all rows with missing elements
k9_data <- k9_data[,1:5409]
missing_rows <- apply(k9_data, 1, function(x) all( x != '?'))
k9_data <- k9_data[missing_rows,]

k9_data <- sapply( k9_data, as.numeric ) # Change inactive and active to 1 and 2
k9_data[,5409][k9_data[,5409]==2] <- -1  # Change all the 2's to -1's

k9_x<-k9_data[,-c(5409)] #Split data into Features(k9_x) and Classification(k9_y)
k9_y<-k9_data[,5409] 

partition1<-createDataPartition(k9_data[,5409], p=.84, list=FALSE) #Split the data into a training/testing(84/16) split

training_x<-k9_x[partition1,]
training_y<-k9_y[partition1]
testing_x<-k9_x[-partition1,]
testing_y<-k9_y[-partition1]

partition2<-createDataPartition(training_y, p=.80, list=FALSE) #Split the training set into training/validation(80/20) split

training_x<-training_x[partition2,]
training_y<-training_y[partition2]
validation_x<-training_x[-partition2,]
validation_y<-training_y[-partition2]

#The three lambdas we tested out
lambdas<-c(.005, .01, .5)
best_a<-NA
best_b<-NA
best_accuracy<-0
best_lambda<-.01
m<-1
n<-50

#Keep track of the accuracy for each lambda for every season
lambda_accuracy_lists<-vector("list",length=length(lambdas))

for (i in 1:length(lambdas)) {
  lambda<-lambdas[i]
  # Initialie a and b randomly
  a<-as.vector(runif(n = 5408, min = 1, max = 10))
  b<-sample(1:10, 1)
  season_count<-100
  
  #vectors to keep track of the accuracy at the end of each season
  season_accuracies<-vector(,season_count)
  
  for (season in 1:season_count) {    # Find best a and b in 100 seasons
    eta<-(m/(season + n))   #Calculate the eta for this season
    steps<-200
    for (step in 1:steps) {
      k<-sample(1:nrow(training_x), 1)    #Every step, sample a random row to update a and b with
      xk<-training_x[k, ]
      yk<-training_y[k]
      output<-yk * (as.numeric(t(a)%*%xk)+b)
      
      #Update a and b with the gradient and eta
      if(output >= 1) {
        a<-a - eta*(lambda*a)
        b<-b-0
      } else {
        a<-a - eta*(lambda * a - yk*xk) 
        b<-b - eta*(-yk)
      }
    }
    
    # Calculate the accuracy for the season
    results<-apply(validation_x, 1, function(x) sign(as.numeric(t(a)%*%x)+b))
    correct_validation<-results==validation_y
    season_accuracies[season]<-sum(correct_validation)/length(correct_validation)
  }
  
  #using the sign function to get the predicted lables and then comparing them to their actual labels to get the accuracy 
  results<-apply(validation_x, 1, function(x) sign(as.numeric(t(a)%*%x)+b))
  correct_validation<-results==validation_y
  accuracy<-sum(correct_validation)/length(correct_validation)
  
  #choosing the best accuracy
  if(accuracy > best_accuracy) {
    best_a<-a
    best_b<-b
    best_accuracy<-accuracy
    best_lambda<-lambda
  }
  
  lambda_accuracy_lists[[i]]<-season_accuracies
}

#using the sign function to get the predicted lables and then comparing them to their actual labels to get the accuracy 
overall_results<-apply(testing_x, 1, function(x) sign(as.numeric(t(best_a)%*%x)+best_b))
correct_testing<-overall_results==testing_y
accuracy<-sum(correct_testing)/length(correct_testing)
print(accuracy)
print(best_lambda)
print(lambda_accuracy_lists)


#from http://www.harding.edu/fmccown/r/ to plot the graphs
#plotting the first graph
g_range <- range(0, 1)
l1<-unlist(lambda_accuracy_lists[1])
l2<-unlist(lambda_accuracy_lists[2])
l3<-unlist(lambda_accuracy_lists[3])
x<- seq(1, 100, 1)
xaxis<- seq(0, 100, 10)
yaxis<- seq(0, 1, .1)
plot(x, l1, type="l", col="blue", ylim=g_range, axes=FALSE, ann=FALSE)
axis(1, at=xaxis)
axis(2, at=yaxis)
box()
lines(x, l2, type="l", pch=22, lty=2, col="red")
lines(x, l3, type="l", pch=22, lty=2, col="green")
title(main="Lambda Accuracies over Seasons", col.main="red", font.main=4)
title(xlab="Seasons", col.lab=rgb(0,0.5,0))
title(ylab="Accuracy", col.lab=rgb(0,0.5,0))