wdat<-read.csv('pima-indians-diabetes.data.txt')
library(klaR)
library(caret)


bigx<-wdat[,-c(9)]
bigy<-as.factor(wdat[,9])
print(bigy)
wtd<-createDataPartition(y=bigy, p=.8, list=FALSE)
trax<-bigx[wtd,]
tray<-bigy[wtd]
model<-train(trax, tray, 'nb', trControl=trainControl(method='cv', number=10))
teclasses<-predict(model,newdata=bigx[-wtd,])
confusionMatrix(data=teclasses, bigy[-wtd])
