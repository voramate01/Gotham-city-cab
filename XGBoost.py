import numpy as np 
import pandas as pd 
import xgboost as xgb
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, ElasticNet, Lasso

from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from sklearn.preprocessing import StandardScaler
import pickle

train = pd.read_csv('new_train.csv', header=0, sep=',')
test = pd.read_csv('new_test.csv', header = 0, sep=',')

train = train.drop(columns = ['pickup_datetime','pickup_weekdays','pickup_weekend'], axis=1)
test = test.drop(columns = ['pickup_datetime','pickup_weekdays','pickup_weekend'], axis=1)

coordinates = np.vstack((test[['pickup_x', 'pickup_y']].values, test[['dropoff_x', 'dropoff_y']].values))
coordinates2 = np.vstack((test[['pickup_x', 'pickup_y']].values, test[['dropoff_x', 'dropoff_y']].values))

scaler = StandardScaler()
coordinates_std = scaler.fit_transform(coordinates)
coordinates_std2 = scaler.fit_transform(coordinates2)

clustering = MiniBatchKMeans(n_clusters=70, random_state=203, batch_size=10000)
model = clustering.fit(coordinates_std)
model2 = clustering.fit(coordinates_std2)
train['kms_pick_cluster'] = pd.Series(model2.predict(scaler.fit_transform(train[['pickup_x', 'pickup_y']])))
train['kms_drop_cluster'] = pd.Series(model2.predict(scaler.fit_transform(train[['dropoff_x', 'dropoff_y']])))
test['kms_pick_cluster'] = pd.Series(model.predict(scaler.fit_transform(test[['pickup_x', 'pickup_y']])))
test['kms_drop_cluster'] = pd.Series(model.predict(scaler.fit_transform(test[['dropoff_x', 'dropoff_y']])))

print(train.head())
print(test.head())

y = train['duration']
x = train.drop(['duration'], axis = 1)
data_dmatrix = xgb.DMatrix(data=x,label=y)

#Train Test split and normalizing the independent features
train_X, test_X, train_Y, test_Y = train_test_split(x, y, test_size = 0.2, shuffle=True, random_state = 123)
scaler1 = preprocessing.StandardScaler().fit(train_X)
train_X = scaler1.transform(train_X)
test_X = scaler1.transform(test_X)

scaler2 = preprocessing.StandardScaler().fit(test)
test_val = scaler2.transform(test)

print("Shape of X train", train_X.shape)
print("Shape of Y train", train_Y.shape)
print("Shape of X test", test_val.shape)


# fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharex=True, sharey=True)
# axes[0].scatter(X_Tr['pickup_x'], X_Tr['pickup_y'], c=X_Tr['kms_pick_cluster'], alpha=0.3, lw = 0, s=20, cmap='Spectral')
# axes[0].set_title('Pickup Locations Cluster')
# axes[0].set_xlabel('Pickup Latitude')
# axes[0].set_ylabel('Pickup Longitude')
# axes[0].set_xlim([-800, 800])
#
# axes[1].scatter(X_Tr['dropoff_x'], X_Tr['dropoff_y'], c=X_Tr['kms_drop_cluster'], alpha=0.3, lw = 0, s=20, cmap='Spectral')
# axes[1].set_title('DropOff Locations Cluster')
# axes[1].set_xlabel('Pickup Latitude')
# axes[1].set_ylabel('Pickup Longitude')
# axes[0].set_xlim([-800, 800])
# plt.show()

xgb_reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.5, gamma=0, importance_type='gain',
       learning_rate=0.1, max_delta_step=0, max_depth=5,
       min_child_weight=1, missing=None, n_estimators=1000, n_jobs=1,
       nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0.03, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=0.6)
#
xgb_reg.fit(train_X,train_Y)
preds = xgb_reg.predict(test_X)

rmse = np.sqrt(mean_squared_error(test_Y, preds))
print("RMSE: %f" % (rmse))
params = { "colsample_bytree":0.5,
       "learning_rate":0.1, "max_depth":5, "objective":"reg:linear",
       "reg_alpha":0.03}


cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=10,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

print((cv_results["test-rmse-mean"]).tail(1))


preds_test = xgb_reg.predict(test_val)

df = pd.DataFrame(preds_test)
df.to_csv('prediction.csv')

# This step was taken to find the optimal estimator and parameter to train XGBoost
params={
     'max_depth': [5],
     'subsample': [0.6],
     'colsample_bytree': [0.5],
     'n_estimators': [1000],
     'reg_alpha': [0.03]
 }


rs = GridSearchCV(xgb_reg,
                   params,
                   cv=5,
                   scoring="neg_mean_squared_error",
                   n_jobs=1,
                   verbose=2)
rs.fit(train_X, train_Y)
best_est = rs.best_estimator_
print(best_est)
print(rs.best_params_)
