
library(factoextra)
library(ISLR)
library(glmnet)
library(pls)
library(lubridate)
library(data.table)
library(pacman)
library(readr)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(caret)
library(stringr)


cab <- read.csv("Train.csv")
head(cab)
str(cab)

#Derive feature from Date
#as.POSIXct(cab$pickup_datetime, tz=Sys.timezone())
#as.POSIXct(strptime(toString(cab$pickup_datetime), "%Y-%m-%d %H:%M:%S"))

cab$pickup_hour <- hour(cab$pickup_datetime)
cab$pickup_week <- week(cab$pickup_datetime)
cab$pickup_month <- month(cab$pickup_datetime)



#plot show several outlier
plot(cab$pickup_x,cab$pickup_y) 
plot(cab$dropoff_x,cab$dropoff_y)
boxplot(cab$pickup_x)
summary(cab)


#clamp transformation
myclip <- function(x, a, b) {
  a + (x-a > 0)*(x-a) - (x-b > 0)*(x-b)
}#input of this function :(vector,min ,max)
cab$pickup_x=myclip( cab$pickup_x, mean(cab$pickup_x)-2.5*sd(cab$pickup_x) , mean(cab$pickup_x)+2.5*sd(cab$pickup_x) )
cab$pickup_y=myclip( cab$pickup_y, mean(cab$pickup_y)-2.5*sd(cab$pickup_y) , mean(cab$pickup_y)+2.5*sd(cab$pickup_y) )
cab$dropoff_x=myclip( cab$dropoff_x, mean(cab$dropoff_x)-2.5*sd(cab$dropoff_x) , mean(cab$dropoff_x)+2.5*sd(cab$dropoff_x) )
cab$dropoff_y=myclip( cab$dropoff_y, mean(cab$dropoff_y)-2.5*sd(cab$dropoff_y) , mean(cab$dropoff_y)+2.5*sd(cab$dropoff_y) )




#Create "euclidean distance" feature
cab["distance"]=  sqrt((cab[,"pickup_x"]- cab[,"dropoff_x"])^2 + (cab[,"pickup_y"] - cab[,"dropoff_y"])^2 )
cab<-cab[,-1]

#Clustering
y_all=c(cab$pickup_y,cab$dropoff_y)#combine both drop off and pickup to get all possible location for clustering
x_all=c(cab$pickup_x,cab$dropoff_x)
forcluster=cbind(x_all,y_all)
plot(x_all,y_all)

k2 <- kmeans(forcluster, centers = 2, nstart = 25)
k3 <- kmeans(forcluster, centers = 3, nstart = 25)
k4 <- kmeans(forcluster, centers = 4, nstart = 25)
k5 <- kmeans(forcluster, centers = 5, nstart =25)
k10 <- kmeans(forcluster, centers = 10, nstart =25)
k25 <- kmeans(forcluster, centers = 25, nstart =25)
k50 <- kmeans(forcluster, centers = 50, nstart =25)
k80 <- kmeans(forcluster, centers = 80, nstart =25)
k100 <- kmeans(forcluster, centers = 100, nstart =25)
k200 <- kmeans(forcluster, centers = 200, nstart =25)
plot(forcluster, col=(k5$cluster+1), main="K-Means Clustering Results with K=5", xlab="", ylab="", pch=20, cex=2)

#Add cluster as a feature
cab$cluster_2 = factor(k2$cluster[1:1300000]) 
cab$cluster_3 = factor(k3$cluster[1:1300000])
cab$cluster_4 = factor(k4$cluster[1:1300000]) 
cab$cluster_5 = k5$cluster[1:1300000] 
cab$cluster_10 = k10$cluster[1:1300000] 
cab$cluster_25 = k25$cluster[1:1300000] 
cab$cluster_50 = k50$cluster[1:1300000] 
cab$cluster_80 = k80$cluster[1:1300000]
cab$cluster_100 = k100$cluster[1:1300000] 
cab$cluster_200 = k200$cluster[1:1300000] 


#Split data randomly to 70:30
nrow(cab)#1,300,000
rand.ind = sample(seq(1:nrow(cab)),nrow(cab))
cab.train = cab[rand.ind[1:910000],]
cab.test = cab[rand.ind[910001:1300000],]
nrow(cab.test)

###Linear regression Model ###
lm.fit <- lm(duration~ . ,data=cab.train)#lm require x as Dataframe
lm.pred =predict(lm.fit , cab.test ,type="response")
lm.mse = sqrt(mean((lm.pred-cab.test$duration)^2))
lm.mse
summary(lm.fit)

###Ridge regression ###
y=cab.train$duration
cab.train.x=model.matrix(duration~., cab.train)[,-1]
grid=10^seq(10,-2,length=100)
ridge.mod=glmnet(cab.train.x, y ,alpha=0,lambda=grid)
cv.out=cv.glmnet(cab.train.x, y ,alpha=0)
plot(cv.out)
bestlam=cv.out$lambda.min
print(log(bestlam))
ridge.pred=predict(ridge.mod,s=bestlam,newx=as.matrix(cab.test[,-1]))
print(sqrt(mean((ridge.pred-cab.test$duration)^2)))

###Lasso ###
grid=10^seq(10,-2,length=100)
lasso.mod=glmnet(cab.train.x,y,alpha=1,lambda=grid)
cv.out=cv.glmnet(cab.train.x,y ,alpha=1)
plot(cv.out)
bestlam=cv.out$lambda.min
print(log(bestlam))
lasso.pred=predict(lasso.mod,s=bestlam, newx=as.matrix(cab.test[,-1]))
print(sqrt(mean((lasso.pred-cab.test$duration)^2)))
#check lasso coeff 
lasso.mod.full=glmnet(as.matrix(cab.train[,-1]),cab.train$duration ,alpha=1,lambda=grid)
lasso.coef=predict(lasso.mod.full,type="coefficients",s=bestlam)[1:9,]
print(lasso.coef) #pickup_hour and pickup_week are dropped by Lasso

#Remove column that are dropped by Lasso
cab.train <- subset(cab.train, select = -c(pickup_hour, pickup_week))
cab.test <- subset(cab.test, select = -c(pickup_hour, pickup_week))

#Save to CSV file
write.csv(cab, file = "xg2.csv",row.names=FALSE)

#For saving the dataframe at a certain point so that we dont have to rerun the code from the begining
save(cab, file= "cab.rda") 
load("cab.rda")

