library(glmnet)

#Read in the data and the identities
data <- read.table('datasets/gene_data.txt', header=FALSE, sep=" ")
identities <- read.table('datasets/gene_identities.txt', header=FALSE)

#Convert to a the data to a matrix, and the identities to a vector. 
# For the identities, the numbers correspond to patients, 
# a positive sign to a normal tissue, and a negative sign to a tumor tissue.
#We only care about the sign
data <- as.matrix(t(data))
identities <- as.vector(sapply(identities, function(x) sign(x)))

#Output the Lasso Regularized Regression plot of misclassification error
png(filename="5.3.longitude.lasso.misclassification.out.png")
lasso.cvfit = cv.glmnet(data, identities, type.measure = "class", nfolds = 3, alpha=1, family="binomial")
plot(lasso.cvfit)
device_out <- dev.off()

print(lasso.cvfit$lambda.min)
print(coef(lasso.cvfit))

#Output the Lasso Regularized Regression plot of deviance error
png(filename="5.3.longitude.lasso.deviance.out.png")
lasso.cvfit = cv.glmnet(data, identities, type.measure = "deviance", nfolds = 3, alpha=1, family="binomial")
plot(lasso.cvfit)
device_out <- dev.off()

# print(lasso.cvfit)

#Output the Lasso Regularized Regression plot of auc error
png(filename="5.3.longitude.lasso.auc.out.png")
lasso.cvfit = cv.glmnet(data, identities, type.measure = "auc", nfolds = 3, alpha=1, family="binomial")
plot(lasso.cvfit)
device_out <- dev.off()

# print(lasso.cvfit)

