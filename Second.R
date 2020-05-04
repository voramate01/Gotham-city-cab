rm(list = ls()) 
graphics.off()

suppressMessages({
  
  knitr::opts_chunk$set(message=F)
  library(data.table)
  library(pacman)
  library(geosphere)
  library(lubridate)
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(gridExtra)
  library(caret)
  library(corrplot)
  library(leaflet)
  library(stringr)
  library(glmnet)
  
  train = read_csv("Train.csv")
  
})

print(head(train))

#Derive feature from column " pickup_datetime"
train$pickup_hour <- hour(train$pickup_datetime)
train$pickup_week <- week(train$pickup_datetime)
train$pickup_month <- month(train$pickup_datetime)
train$pickup_weekdays <- weekdays(train$pickup_datetime)
train$pickup_weekend <- ifelse(train$pickup_weekdays==1 | train$pickup_weekdays==7,"Weekend","not-Weekend")

train = as.data.table(train)

train[,pickup_datetime:=as.Date(pickup_datetime)]

train[,":="(
  pickup_yday=yday(pickup_datetime)
  ,pickup_mday=mday(pickup_datetime)
)]
print(head(train))

#Plot mean trip duration over time and Total rides overtime
train %>% 
  ggplot(aes(x=duration)) + 
  geom_histogram(bins=40000, fill="red")+
  theme_bw()+theme(axis.title = element_text(size=12),axis.text = element_text(size=12))+
  ylab("Density")+coord_cartesian(x=c(0,6000))

plot1 = train[, list(mean_trip_duration= mean(duration)), by=pickup_datetime] %>%
  ggplot(aes(x=pickup_datetime, y=mean_trip_duration)) + 
  geom_bar(stat='identity', fill='steelblue') + 
  labs(x='', y='Mean Trip Duration', title='Mean Trip Duration over time')

plot2 = train[, .N, by=pickup_datetime] %>%
  ggplot(aes(x=pickup_datetime, y=N)) + 
  geom_bar(stat='identity', fill='steelblue') + 
  labs(x='', y='Number of Rides', title='Rides over time')

grid.arrange(plot1, plot2, ncol =2)


#Plot Mean trip duration by weekdays
plot3 <-train[, list(mean_trip_duration = mean(duration)), by = pickup_weekdays] %>%
  ggplot(aes(x = pickup_weekdays, y = mean_trip_duration)) +
  geom_bar(stat = 'identity', fill = 'steelblue') +
  labs(x = 'Month', y = 'Mean Trip Duration', title = 'Mean Trip duration by weekdays')

grid.arrange(plot3)

#Plot mean trip duration by hour and Total ride per hour
plot1 <-train[, list(mean_trip_duration = mean(duration)), by = pickup_hour] %>%
  ggplot(aes(x = as.factor(pickup_hour), y = mean_trip_duration)) +
  geom_bar(stat = 'identity', fill = 'steelblue') +
  labs(x = 'Hours', y = 'Mean Trip Duration', title = 'Mean Trip duration by hour of the day')

plot2 = train[,.N, by=pickup_hour] %>%
  ggplot(aes(x=pickup_hour, y=N)) + 
  geom_bar(stat='identity', fill='steelblue') + 
  labs(x='', y='Number of Rides', title='Total Rides Per Hour')

grid.arrange(plot1, plot2, ncol =2)



#Derive "Distance" feature from pickup and dropoff x-y coordinate.
is.data.table(train)
train = as.data.table(train)

train <- train[,Distance := 
                 sqrt((train[,'pickup_x']-train[,'dropoff_x'])^2 + (train[,'pickup_y']-train[,'dropoff_y'])^2)
                               ]
train <- filter(train, duration != 0)
print(head(train))

#Plot Distance vs density
train %>% 
  ggplot(aes(x=Distance)) + 
  geom_histogram(bins=4000, fill="red")+
  theme_bw()+theme(axis.title = element_text(size=11),axis.text = element_text(size=8))+
  ylab("Density")+coord_cartesian(x=c(0,25))

#Create "speed" column
train[,c("speed")]=(train$Distance/(train$duration/3600))

#Plot speed vs density
train %>% 
  ggplot(aes(x=speed)) + 
  geom_histogram(bins=5000, fill="red")+
  theme_bw()+theme(axis.title = element_text(size=11),axis.text = element_text(size=8))+
  ylab("Density")+coord_cartesian(x=c(0,100))

#Check missing value
sum(is.na(train))

summary(train$speed)
library('ggplot2') # visualisation
library('scales') # visualisation
library('grid') # visualisation
library('RColorBrewer') # visualisation
library('corrplot') # visualisation
library('alluvial') # visualisation
library('dplyr') # data manipulation

#Plot Median speed vs Day of the week
p1 <- train %>%
  
  group_by(pickup_weekdays) %>%
  
  summarise(median_speed = median(speed)) %>%
  
  ggplot(aes(pickup_weekdays, median_speed)) +
  
  geom_point(size = 4) +
  
  labs(x = "Day of the week", y = "Median speed [km/h]")

#Plot Median speed vs Hour of the day
p2 <- train %>%
  
  group_by(pickup_hour) %>%
  
  summarise(median_speed = median(speed)) %>%
  
  ggplot(aes(pickup_hour, median_speed)) +
  
  geom_smooth(method = "loess", span = 1/2) +
  
  geom_point(size = 4) +
  
  labs(x = "Hour of the day", y = "Median speed [km/h]") +
  
  theme(legend.position = "none")

#Heatmap median speed vs hour of the day & Day of the week
p3 <- train %>%
  
  group_by(pickup_weekdays, pickup_hour) %>%
  
  summarise(median_speed = median(speed)) %>%
  
  ggplot(aes(pickup_hour, pickup_weekdays, fill = median_speed)) +
  
  geom_tile() +
  
  labs(x = "Hour of the day", y = "Day of the week") +
  
  scale_fill_distiller(palette = "Spectral")

layout <- matrix(c(1,2,3,3),2,2,byrow=TRUE)
grid.arrange(p1, p2, p3)


# remove extreme trip duration
day_plus_trips <- train %>%
  
  filter(duration > 24*3600)

day_plus_trips %>% select(pickup_datetime, speed)


day_trips <- train %>%
  
  filter(duration < 24*3600 & duration > 22*3600)


min_trips <- train %>%
  
  filter(duration < 5*60)

min_trips %>% 
  
  arrange(Distance) %>%
  
  select(Distance, pickup_datetime, speed, duration) %>%
  
  head(5)

zero_dist <- train %>%
  
  filter(near(Distance,0))

nrow(zero_dist)

# remove outliers

m=mean(train[,'duration'])
s=sd(train[,'duration'])
train=train[train$duration <=(m+2*s),]
train=train[train$duration >=(m-2*s),]

#Remove speed column (speed will not be used for model training, it's used for visualization only)
head(train)
train=subset(train, select= -c(speed))

#plot pick up and drop of locations --> see so many Outliers
cab<-train
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
#plot k mean cluster with 5 centroid
plot(forcluster, col=(k5$cluster+1), main="K-Means Clustering Results with K=5", xlab="", ylab="", pch=20, cex=2)

#Add cluster as features
cab$cluster_2 = k2$cluster[1:1241775] 
cab$cluster_3 = k3$cluster[1:1241775]
cab$cluster_4 = k4$cluster[1:1241775] 
cab$cluster_5 = k5$cluster[1:1241775] 
cab$cluster_10 = k10$cluster[1:1241775] 
cab$cluster_25 = k25$cluster[1:1241775] 
cab$cluster_50 = k50$cluster[1:1241775] 
cab$cluster_80 = k80$cluster[1:1241775]
cab$cluster_100 = k100$cluster[1:1241775] 
cab$cluster_200 = k200$cluster[1:1241775] 


#Split data randomly 70:30
#cab<-na.omit(cab)
nrow(cab)#1,241775
rand.ind = sample(seq(1:nrow(cab)),nrow(cab))
cab.train = cab[rand.ind[1:869243],]
cab.test = cab[rand.ind[869244:1241775],]
nrow(cab.test)
cab.train.x= cab.train[,-c(1,2,10,11)]
cab.test.x= cab.train[,-c(1,2,10,11)]
head(cab.train.x)


###Ridge regression ###
y=cab.train$duration
grid=10^seq(10,-2,length=100)
ridge.mod=glmnet(as.matrix(cab.train.x), y ,alpha=0,lambda=grid)
cv.out=cv.glmnet(as.matrix(cab.train.x), y ,alpha=0)
plot(cv.out)
bestlam=cv.out$lambda.min
print(log(bestlam))
ridge.pred=predict(ridge.mod,s=bestlam,newx=as.matrix(cab.test.x))
print(sqrt(mean((ridge.pred-cab.test$duration)^2)))

###Lasso ###
grid=10^seq(10,-2,length=100)
lasso.mod=glmnet(as.matrix(cab.train.x),y,alpha=1,lambda=grid)
cv.out=cv.glmnet(as.matrix(cab.train.x),y ,alpha=1)
plot(cv.out)
bestlam=cv.out$lambda.min
print(log(bestlam))
lasso.pred=predict(lasso.mod,s=bestlam, newx=as.matrix(cab.test.x))
print(sqrt(mean((lasso.pred-cab.test$duration)^2)))
#check lasso coeff 
lasso.mod.full=glmnet(as.matrix(cab.train.x),y ,alpha=1,lambda=grid)
lasso.coef=predict(lasso.mod.full,type="coefficients",s=bestlam)[1:12,]
print(lasso.coef)

#sAVE to CSV file
write.csv(cab, file = "new_train.csv",row.names=FALSE)



