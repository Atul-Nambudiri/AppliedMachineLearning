library(glmnet)

#Read in the data
col_names <- c("Gender", "Length", "Diameter", "Height", "Whole_w", "Shucked_w", "Viscera_w", "Shell_w", "Age")
data <- read.csv('datasets/abalone.data', col.names=col_names, stringsAsFactors=FALSE)

#The last column in the dataset is actually Rings. Add 1.5 to this to get the Age
data[,9] = data[, 9] + 1.5

#Create the linear regression
res<-lm(Age~Length + Diameter + Height + Whole_w + Shucked_w + Viscera_w + Shell_w, data=data)

#Output the fitted vs residuals plot
png(filename="7.11a.out.png")
plot(fitted(res), resid(res), type="p", main="Without Gender - 7.11a", xlab="Fitted", ylab="Residuals")
abline(0, 0)
device_out <- dev.off()

print("R-Squared")
print(summary(res)$r.squared)

#Output the Cross Validation Error Plot
png(filename="7.11a.cv.out.png")
x <- as.matrix(data[,2:8])
y <- as.vector(data[,9])
cvfit = cv.glmnet(x, y)
plot(cvfit)
device_out <- dev.off()


#Output the Regularized Regression using the lambda with the lowest mean error
png(filename="7.11a.reg.out.png")
res = predict(cvfit, newx = x, s = "lambda.min")
plot(res, y-res, type="p", main="Regularized Without Gender - 7.11a", xlab="Fitted", ylab="Residuals")
abline(0, 0)
device_out <- dev.off()


print("Lambda with lowest mean Error:")
print(cvfit$lambda.min)