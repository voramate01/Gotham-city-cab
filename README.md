# Gotham-city-cab-

Gotham City Cab project, we aim to study the pickup time and location data to predict the duration of a cab ride. 
Our initial observation of the data tells us that we may need to consider using the x,y coordinate data to derive 
useful feature such as ‘distance’ which may play an important role in predicting the duration of cab rides. 
We also consider clustering method to group the pick-up point and drop-off point so that we can get a better understanding 
of cab duration affected by certain location of Gotham City. Lastly for modeling, we consider Lasso, Ridge, 
and Extreme Gradient Boosting method to predict cab duration.

Problem Analysis
- The data has 1.3 milion observation with 6 variables. The response variable is duration.
- There are one datetime variable(pickup_datetime) and 5 numeric variables
- Acknowledge that pickup_x, pickup_y, dropoff_x, dropoff_y represents location coordinate data
