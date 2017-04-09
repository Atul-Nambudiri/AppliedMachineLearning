library(glmnet)

#Read in the data
col_names <- c("Gender", "Length", "Diameter", "Height", "Whole_w", "Shucked_w", "Viscera_w", "Shell_w", "Age")
data <- read.csv('datasets/abalone.data', col.names=col_names, stringsAsFactors=FALSE)

#The last column in the dataset is actually Rings. Add 1.5 to this to get the Age
data[,9] = data[, 9] + 1.5

#Change gender from {M,F,I} to {1, 0, -1}
data[,1][data[,1] == "M"] <- 1 
data[,1][data[,1] == "F"] <- 0 
data[,1][data[,1] == "I"] <- -1 

data$Gender <- as.numeric(data$Gender)

#Create the linear regression
res<-lm(Age~Gender + Length + Diameter + Height + Whole_w + Shucked_w + Viscera_w + Shell_w, data=data)

#Output the fitted vs residuals plot
png(filename="7.11b.out.png")
plot(fitted(res), resid(res), type="p", main="With Gender - 7.11b", xlab="Fitted", ylab="Residuals")
abline(0, 0)
device_out <- dev.off()

print("R-Squared")
print(summary(res)$r.squared)

#Output the Regularized Regression using the lambda with the lowest mean error
png(filename="7.11b.cv.out.png")
x <- as.matrix(data[,1:8])
y <- as.vector(data[,9])
cvfit = cv.glmnet(x, y)
plot(cvfit)
device_out <- dev.off()


#Output the Regularized Regression using the lambda with the lowest mean error
png(filename="7.11b.reg.out.png")
res = predict(cvfit, newx = x, s = "lambda.min")
plot(res, y-res, type="p", main="Regularized With Gender - 7.11b", xlab="Fitted", ylab="Residuals")
abline(0, 0)
device_out <- dev.off()


print("Lambda with lowest mean Error:")
print(cvfit$lambda.min)