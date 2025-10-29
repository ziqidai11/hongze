'''


########################   网格搜索  #########################################
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'max_depth': [3, 5, 7, 9, 12],
    'n_estimators': [100, 300, 500, 800, 1000],
    'min_child_weight': [1, 3, 5, 7, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3, 0.5, 1],
    'reg_alpha': [0, 0.1, 0.5, 1],
    'reg_lambda': [1, 2, 5, 10]
}

xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=2,
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

print("Best Parameters:", grid_search.best_params_)


########################   网格搜索  #########################################
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'max_depth': [3, 5, 7, 9, 12],
    'n_estimators': [100, 300, 500, 800, 1000],
    'min_child_weight': [1, 3, 5, 7, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3, 0.5, 1],
    'reg_alpha': [0, 0.1, 0.5, 1],
    'reg_lambda': [1, 2, 5, 10]
}

best_score = float('inf')
best_params = {}

for params in ParameterGrid(param_grid):
    xgb_model = XGBRegressor(**params)
    xgb_model.fit(X_train_scaled, y_train)
    predictions = xgb_model.predict(X_test_scaled)
    score = mean_squared_error(y_test, predictions)
    if score < best_score:
        best_score = score
        best_params = params

print("Best Parameters:", best_params)


########################   随机搜索  #########################################
param_distributions = {
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'max_depth': [3, 5, 7, 9, 12],
    'n_estimators': [100, 300, 500, 800, 1000], 
    'min_child_weight': [1, 3, 5, 7, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3, 0.5, 1],
    'reg_alpha': [0, 0.1, 0.5, 1],
    'reg_lambda': [1, 2, 5, 10]
}

xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=100,  # 随机搜索的迭代次数
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_scaled, y_train)

print("最佳参数:", random_search.best_params_)
print("最佳得分:", -random_search.best_score_)  # 转换回MSE

'''


'''

########################   贝叶斯优化  #########################################
from bayes_opt import BayesianOptimization

def xgb_eval(learning_rate, max_depth, n_estimators, min_child_weight, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda):
    xgb_model = XGBRegressor(
        objective='reg:squarederror',
        learning_rate=learning_rate,
        max_depth=int(max_depth),
        n_estimators=int(n_estimators),
        min_child_weight=int(min_child_weight),
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=42
    )
    xgb_model.fit(X_train_scaled, y_train)
    predictions = xgb_model.predict(X_test_scaled)
    return -mean_squared_error(y_test, predictions)

# 定义参数范围
params = {
    'learning_rate': (0.01, 0.2),
    'max_depth': (3, 12),
    'n_estimators': (100, 1000),
    'min_child_weight': (1, 10),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'gamma': (0, 2),
    'reg_alpha': (0, 2),
    'reg_lambda': (1, 10)
}

optimizer = BayesianOptimization(f=xgb_eval, pbounds=params, random_state=42, verbose=2)
optimizer.maximize(init_points=20, n_iter=100)

print("Best Parameters:", optimizer.max)

'''