library(glmnet)

#Read in the data
col_names <- c("Gender", "Length", "Diameter", "Height", "Whole_w", "Shucked_w", "Viscera_w", "Shell_w", "Age")
data <- read.csv('datasets/abalone.data', col.names=col_names, stringsAsFactors=FALSE)

#The last column in the dataset is actually Rings. Add 1.5 to this to get the Age. Then take the log of it
data[,9] = log(data[, 9] + 1.5)

#Create the linear regression
res<-lm(Age~Length + Diameter + Height + Whole_w + Shucked_w + Viscera_w + Shell_w, data=data)

#Output the fitted vs residuals plot
png(filename="7.11c.out.png")

#Apply the exp function to undo the log on Age
newFittedAge = exp(fitted(res))
newResid = exp(data$Age) - newFittedAge

plot(newFittedAge, newResid, type="p", main="Log Age Without Gender - 7.11c", xlab="Fitted", ylab="Residuals")
abline(0, 0)
device_out <- dev.off()

print("R-Squared")
print(summary(res)$r.squared)

#Output the Cross Validation Error Plot
png(filename="7.11c.cv.out.png")
x <- as.matrix(data[,2:8])
y <- as.vector(data[,9])
cvfit = cv.glmnet(x, y)
plot(cvfit)
device_out <- dev.off()


#Output the Regularized Regression using the lambda with the lowest mean error
png(filename="7.11c.reg.out.png")
res = predict(cvfit, newx = x, s = "lambda.min")
plot(exp(res), exp(y)-exp(res), type="p", main="Regularized Log Age Without Gender - 7.11c", xlab="Fitted", ylab="Residuals")
abline(0, 0)
device_out <- dev.off()


print("Lambda with lowest mean Error:")
print(cvfit$lambda.min)