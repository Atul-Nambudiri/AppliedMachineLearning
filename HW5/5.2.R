library(glmnet)
library(MASS)
require(gdata)

df = read.xls ("datasets/default_of_credit_card_clients.xls", sheet = 1, header = TRUE, pattern = 'ID')
features <- as.matrix(df[,2:24])
default_payment <- as.vector(df[,25])

#Output the Ridge regression Cross Validation Error Plot for default_payment
png(filename="5.2.payment.ridge.out.png")
ridge.cvfit = cv.glmnet(features, default_payment, alpha=0, family="binomial")
plot(ridge.cvfit)
device_out <- dev.off()

#Output the Ridge Regularized Regression using the lambda with the lowest binomal deviance
res_ridge = predict(ridge.cvfit, newx = features, s = "lambda.min", type= "class")
res_ridge = as.numeric(res_ridge)
ans_ridge = res_ridge == default_payment
accuracy_ridge = sum(ans_ridge)/length(res_ridge)

cat("Ridge Regularized Lambda with Lowest Binomial Deviance:", ridge.cvfit$lambda.min, "\n")
cat("Accuracy for Ridge is ", accuracy_ridge, "\n")

#Output the Lasso regression Cross Validation Error Plot for default_payment
png(filename="5.2.payment.lass.out.png")
lasso.cvfit = cv.glmnet(features, default_payment, alpha=1, family="binomial")
plot(lasso.cvfit)
device_out <- dev.off()

#Output the Lasso Regularized Regression using the lambda with the lowest binomal deviance
res_lasso = predict(lasso.cvfit, newx = features, s = "lambda.min", type= "class")
res_lasso = as.numeric(res_lasso)
ans_lasso = res_lasso == default_payment
accuracy_lasso = sum(ans_lasso)/length(res_lasso)

cat("Lasso Regularized Lambda with Lowest Binomial Deviance:", lasso.cvfit$lambda.min, "\n")
cat("Accuracy for Lasso is ", accuracy_lasso, "\n")


#Output the Elastic Net regression Cross Validation Error Plot for default_payment
png(filename="5.2.payment.elastic.out.png")
elnet.cvfit = cv.glmnet(features, default_payment, alpha=0.5, family="binomial")
plot(elnet.cvfit)
device_out <- dev.off()

#Output the Elastic Net Regularized Regression using the lambda with the lowest binomal deviance
res_elnet = predict(elnet.cvfit, newx = features, s = "lambda.min", type= "class")
res_elnet = as.numeric(res_elnet)
ans_elnet = res_elnet == default_payment
accuracy_elnet = sum(ans_elnet)/length(res_elnet)

cat("Elastic Net Regularized Lambda with Lowest Binomial Deviance:", elnet.cvfit$lambda.min, "\n")
cat("Accuracy for Elastic Net is ", accuracy_elnet, "\n")
