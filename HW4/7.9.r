# 7.9a -----------------------------------------------------------------------

efd <- read.table('datasets/brunhild.txt', header=TRUE)

# Creating new columns
hours <- stack(efd, select=1)
sulfate <- stack(efd, select=2)

#Creating new data frame and doing log of both variables
foo<-data.frame(Hours=hours[,c("values")],
                Sulfate=sulfate[,c("values")])
foo <- log(foo)

#Plotting data framea and create a regression line
foo.lm <- lm(Sulfate~Hours, data=foo)
plot(foo)
abline(foo.lm)


# 7.9b--------------------------------------------------------------------------------------

library('MASS')

efd <- read.table('datasets/brunhild.txt', header=TRUE)

# Creating new columns
hours <- stack(efd, select=1)
sulfate <- stack(efd, select=2)

#Creating new data frame 
foo<-data.frame(Hours=hours[,c("values")],
                Sulfate=sulfate[,c("values")])


#Plotting data framea and create a regression line
foo.lm <- lm(foo$Sulfate~foo$Hours + I(log(foo$Hours)), data=foo)
plot(foo)
lines(sort(foo$Hours), fitted(foo.lm)[order(foo$Hours)], col='red', type='b') 

# 7.9c --------------------------------------------------------------------------------------

efd <- read.table('datasets/brunhild.txt', header=TRUE)

# Creating new columns
hours <- stack(efd, select=1)
sulfate <- stack(efd, select=2)

#Creating new data frame and doing log of both variables
foo<-data.frame(Hours=hours[,c("values")],
                Sulfate=sulfate[,c("values")])
foo <- log(foo)

#Plotting data framea and create a regression line
foo.lm <- lm(Sulfate~Hours, data=foo)
foo.res = resid(foo.lm)

plot(foo$Hours, foo.res)
abline(0,0)

# 7.9d -------------------------------------------------------------------------------------------

#The residual plots shows a slight pattern so our regression line is pretty good for the log log plot
#and the orginal plot
