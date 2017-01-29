wdat<-read.csv('pima-indians-diabetes.data.txt')
library(klaR)
library(caret)
bigx<-wdat[,-c(9)]
bigy<-as.factor(wdat[,9])
wtd<-createDataPartition(y=bigy, p=.8, list=FALSE)
svm<-svmlight(bigx[wtd,], bigy[wtd])
labels<-predict(svm, bigx[-wtd,])
foo<-labels$class
sum(foo==bigy[-wtd])/(sum(foo==bigy[-wtd])+sum(!(foo==bigy[-wtd])))