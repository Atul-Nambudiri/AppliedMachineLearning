library(glmnet)
library(MASS)

#Read in the data
data <- read.csv('datasets/default_plus_chromatic_features_1059_tracks.txt', header=FALSE, stringsAsFactors=FALSE)

#Get a features matrix, and latitude and longitude vectors. We add 360 to the lat/long to get rid of negative values
features <- as.matrix(data[,1:116])
latitude <- as.vector(data[,117]) + 360
longitude <- as.vector(data[,118]) + 360

#Create the latitude linear regression
res<-lm(latitude~features)

#Output the fitted vs residuals plot for latitude
png(filename="5.1.latitude.out.png")

plot(fitted(res), resid(res), type="p", main="Latitude", xlab="Fitted", ylab="Residuals")
abline(0, 0)
device_out <- dev.off()

cat("Latitude R-Squared", summary(res)$r.squared, "\n")

# Use boxcox to find max lambda for latitude
png(filename="5.1.latitude.boxcox.out.png")
b = boxcox(res, lambda = seq(2,14, 1/20))
maxlambda = b$x[which.max(b$y)]
device_out <- dev.off()

cat("Latitude Box Cox Lambda", maxlambda, "\n")
latitude = (latitude ** maxlambda - 1)/maxlambda

res <- lm(latitude~features)

#Output the fitted vs residuals plot for latitude
png(filename="5.1.latitude.boxcox.reg.out.png")

plot(fitted(res), resid(res), type="p", main="Latitude With Boxcox", xlab="Fitted", ylab="Residuals")
abline(0, 0)
device_out <- dev.off()

cat("New Latitude R-Squared", summary(res)$r.squared, "\n")

#Create the longitude linear regression
res<-lm(longitude~features)

#Output the fitted vs residuals plot for longitude
png(filename="5.1.longitude.out.png")

plot(fitted(res), resid(res), type="p", main="Longitude", xlab="Fitted", ylab="Residuals")
abline(0, 0)
device_out <- dev.off()

cat("Longitude R-Squared:", summary(res)$r.squared, "\n")

#Use boxcox to find maxlambda for longitude
png(filename="5.1.longitude.boxcox.out.png")
b <- boxcox(res, lambda = seq(-10,10, 1/20))
maxlambda = b$x[which.max(b$y)]
device_out <- dev.off()

cat("Longitude Box Cox Lambda", maxlambda, "\n")
longitude <- (longitude ** maxlambda - 1)/maxlambda

res <- lm(longitude~features)

#Output the fitted vs residuals plot for longitude
png(filename="5.1.longitude.boxcox.reg.out.png")

plot(fitted(res), resid(res), type="p", main="Longitude With Boxcox", xlab="Fitted", ylab="Residuals")
abline(0, 0)
device_out <- dev.off()

cat("New Longitude R-Squared", summary(res)$r.squared, "\n")











#Output the Ridge regression Cross Validation Error Plot for Latitude
png(filename="5.1.latitude.ridge.out.png")
ridge.cvfit = cv.glmnet(features, latitude, alpha=0)
plot(ridge.cvfit)
device_out <- dev.off()

#Output the Ridge Regularized Regression using the lambda with the lowest mean error
png(filename="5.1.latitude.ridge.reg.out.png")
res = predict(ridge.cvfit, newx = features, s = "lambda.min")
plot(res, latitude-res, type="p", main="Ridge Regularized Latitude", xlab="Fitted", ylab="Residuals")
abline(0, 0)
device_out <- dev.off()

cat("Ridge Regularized Lambda for Latitude with Lowest Mean Error:", ridge.cvfit$lambda.min, "\n")


#Output the Lasso regression Cross Validation Error Plot for Latitude
png(filename="5.1.latitude.lasso.out.png")
lasso.cvfit = cv.glmnet(features, latitude, alpha=1)
plot(lasso.cvfit)
device_out <- dev.off()

#Output the Lasso Regularized Regression using the lambda with the lowest mean error
png(filename="5.1.latitude.lasso.reg.out.png")
res = predict(lasso.cvfit, newx = features, s = "lambda.min")
plot(res, latitude-res, type="p", main="Lasso Regularized Latitude", xlab="Fitted", ylab="Residuals")
abline(0, 0)
device_out <- dev.off()

cat("Lasso Regularized Lambda for Latitude with Lowest Mean Error:", lasso.cvfit$lambda.min, "\n")

# Print out the coefficients(Number of features used) In this case, its 14 variables
# print(coef(lasso.cvfit))

#Output the Ridge Regression Cross Validation Error Plot for Longitude
png(filename="5.1.longitude.ridge.out.png")
ridge.cvfit = cv.glmnet(features, longitude, alpha=0)
plot(ridge.cvfit)
device_out <- dev.off()

#Output the Ridge Regularized Regression using the lambda with the lowest mean error
png(filename="5.1.longitude.ridge.reg.out.png")
res = predict(ridge.cvfit, newx = features, s = "lambda.min")
plot(res, longitude-res, type="p", main="Ridge Regularized Longitude", xlab="Fitted", ylab="Residuals")
abline(0, 0)
device_out <- dev.off()

cat("Ridge Regularized Lambda for Longitude with Lowest Mean Error:", ridge.cvfit$lambda.min, "\n")


#Output the Lasso regression Cross Validation Error Plot for Longitude
png(filename="5.1.longitude.lasso.out.png")
lasso.cvfit = cv.glmnet(features, longitude, alpha=1)
plot(lasso.cvfit)
device_out <- dev.off()

#Output the Lasso Regularized Regression using the lambda with the lowest mean error
png(filename="5.1.longitude.lasso.reg.out.png")
res = predict(lasso.cvfit, newx = features, s = "lambda.min")
plot(res, longitude-res, type="p", main="Lasso Regularized Longitude", xlab="Fitted", ylab="Residuals")
abline(0, 0)
device_out <- dev.off()

# Print out the coefficients(Number of features used) In this case, its 46 variables
# print(coef(lasso.cvfit))

cat("Lasso Regularized Lambda for Longitude with Lowest Mean Error:", lasso.cvfit$lambda.min, "\n")