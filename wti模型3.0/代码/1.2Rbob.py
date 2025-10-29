import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from Dcel import update_excel_data
from Dtool import fill_missing_values, reverse_column

# ------------ 全局配置参数 -----------
FILE_PATH = 'data_input/RBOB.xlsx'
OUTPUT_DAILY = 'eta/RBOB_Daily.xlsx'
OUTPUT_MONTHLY = 'eta/RBOB_Monthly.xlsx'
UPDATE_FILE_PATH = "eta/1.WTI_update_data.xlsx"
UPDATE_SHEET_NAME = "日度数据表"
UPDATE_IDENTIFIER = "RBOB"

NUM_BOOST_ROUND = 1000
RANDOM_STATE = 42
USE_HYPERPARAM_TUNING = False    # 若 False 则直接使用默认参数

TARGET_COL = '美国RBOB汽油裂解'  # 预测目标
TEST_PERIOD = 20                 # 测试集样本数量

SEARCH_MODE = "random"           # 可选 "grid"/"bayesian"/"random"
SHOW_PLOTS = True                # 是否显示最终预测图表

ADJUST_FULL_PREDICTIONS = True

DEFAULT_PARAMS = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.1309,
    'max_depth': 8,
    'min_child_weight': 3,
    'gamma': 2,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'eval_metric': 'rmse',
    'seed': RANDOM_STATE,
    'reg_alpha': 0.45,
    'reg_lambda': 1.29,
}

# —— 因子预处理相关配置  —— 
FILL_METHODS = {
    '美国2年通胀预期': 'rolling_mean_5',
    '美国首次申领失业金人数/4WMA': 'interpolate',
    '道琼斯旅游与休闲/工业平均指数': 'interpolate',
    '美国EIA成品油总库存(预测/供应需求3年季节性)': 'interpolate',
    '美国成品车用汽油倒推产量（预测/汽油库存维持上年季节性）/8WMA': 'interpolate',
    '美国成品车用汽油炼厂与调和装置净产量/4WMA(预测/上年季节性)超季节性/5年': 'interpolate',
    '美国炼厂可用产能（路透）(预测)': 'interpolate',
    '美国炼厂CDU装置检修量（新）': 'interpolate',
    '美湾单位辛烷值价格(预测/季节性)': 'interpolate',
    '美国汽油调和组分RBOB库存(预测/线性外推)超季节性/3年': 'interpolate'
}

SHIFT_CONFIG = [
    ('美国2年通胀预期', 56, '美国2年通胀预期_提前56天'),
    ('美国首次申领失业金人数/4WMA', 100, '美国首次申领失业金人数/4WMA_提前100天'),
    ('美国首次申领失业金人数/4WMA', 112, '美国首次申领失业金人数/4WMA_提前112天'),
    ('道琼斯旅游与休闲/工业平均指数', 14, '道琼斯旅游与休闲/工业平均指数_提前14天'),
    ('美国EIA成品油总库存(预测/供应需求3年季节性)', 15, '美国EIA成品油总库存(预测/供应需求3年季节性)_提前15天'),
    ('美国成品车用汽油炼厂与调和装置净产量/4WMA(预测/上年季节性)超季节性/5年', 30,
     '美国成品车用汽油炼厂与调和装置净产量/4WMA(预测/上年季节性)超季节性/5年_提前30天'),
    ('美国炼厂CDU装置检修量（新）', 30, '美国炼厂CDU装置检修量（新）_提前30天'),
    ('美国炼厂可用产能（路透）(预测)', 100, '美国炼厂可用产能（路透）(预测)_提前100天')
]

REVERSE_CONFIG = [
    ('美国首次申领失业金人数/4WMA', '美国首次申领失业金人数/4WMA_逆序'),
    ('美国首次申领失业金人数/4WMA_提前100天', '美国首次申领失业金人数/4WMA_提前100天_逆序'),
    ('美国首次申领失业金人数/4WMA_提前112天', '美国首次申领失业金人数/4WMA_提前112天_逆序'),
    ('美国EIA成品油总库存(预测/供应需求3年季节性)', '美国EIA成品油总库存(预测/供应需求3年季节性)_逆序'),
    ('美国EIA成品油总库存(预测/供应需求3年季节性)_提前15天', '美国EIA成品油总库存(预测/供应需求3年季节性)_提前15天_逆序'),
    ('美国炼厂可用产能（路透）(预测)_提前100天', '美国炼厂可用产能（路透）(预测)_逆序'),
    ('美国汽油调和组分RBOB库存(预测/线性外推)超季节性/3年', '美国汽油调和组分RBOB库存(预测/线性外推)超季节性/3年_逆序')
]

SPECIAL_REVERSE = {
    '美国汽油调和组分RBOB库存(预测/线性外推)超季节性/3年_逆序_2022-01-01': {
        'base_column': '美国汽油调和组分RBOB库存(预测/线性外推)超季节性/3年_逆序',
        'condition_date': pd.Timestamp('2022-01-01')
    }
}

# ------------ 数据加载与预处理 ------------
def load_and_preprocess_data(file_path):
    excel_data = pd.ExcelFile(file_path)
    df = excel_data.parse('Sheet1')
    df.rename(columns={'DataTime': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.set_index('Date', inplace=True)

    full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df_daily = df.reindex(full_date_range)
    df_daily.reset_index(inplace=True)
    df_daily.rename(columns={'index': 'Date'}, inplace=True)

    df_daily = fill_missing_values(df_daily, FILL_METHODS, return_only_filled=False)
    for col, shift_days, new_col in SHIFT_CONFIG:
        df_daily[new_col] = df_daily[col].shift(shift_days)

    last_valid_idx = df_daily[TARGET_COL].last_valid_index()
    last_day = df_daily['Date'].iloc[last_valid_idx]
    last_day_ext = last_day + pd.Timedelta(days=30)

    df_daily = df_daily[(df_daily['Date'] >= '2009-08-01') & (df_daily['Date'] <= last_day_ext)]
    df_daily = df_daily[df_daily['Date'].dt.dayofweek < 5]
    for base_col, new_col in REVERSE_CONFIG:
        df_daily[new_col] = reverse_column(df_daily, base_col)
    for special_col, config in SPECIAL_REVERSE.items():
        base_col = config['base_column']
        condition_date = config['condition_date']
        df_daily[special_col] = np.where(df_daily['Date'] >= condition_date,
                                         df_daily[base_col],
                                         np.nan)
    df_daily = df_daily[(df_daily['Date'] > last_day) | df_daily[TARGET_COL].notna()]
    return df_daily, last_day

# ------------ 数据划分与特征构建 ------------
def split_and_build_features(df_daily, last_day):
    train_data = df_daily[df_daily['Date'] <= last_day].copy()
    test_data = train_data[-TEST_PERIOD:].copy()
    train_data = train_data[:-TEST_PERIOD].copy()
    future_data = df_daily[df_daily['Date'] > last_day].copy()

    feature_columns = [
        '美湾单位辛烷值价格(预测/季节性)',
        '美国炼厂CDU装置检修量（新）_提前30天', 
        '美国EIA成品油总库存(预测/供应需求3年季节性)_提前15天_逆序', 
        '美国首次申领失业金人数/4WMA_提前100天_逆序',
        '美国成品车用汽油倒推产量（预测/汽油库存维持上年季节性）/8WMA',
        '美国成品车用汽油炼厂与调和装置净产量/4WMA(预测/上年季节性)超季节性/5年_提前30天',
        '美国汽油调和组分RBOB库存(预测/线性外推)超季节性/3年_逆序_2022-01-01'
    ]
    X_train = train_data[feature_columns]
    y_train = train_data[TARGET_COL]
    X_test = test_data[feature_columns]
    y_test = test_data[TARGET_COL]
    X_future = future_data[feature_columns]
    return X_train, y_train, X_test, y_test, X_future, train_data, test_data, future_data

# ------------ 特征缩放与异常值检测 ------------
def scale_and_weight_features(X_train, X_test, X_future):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_future_scaled = scaler.transform(X_future)
    return scaler, X_train_scaled, X_test_scaled, X_future_scaled

def detect_outliers_weights(X, weight_normal=1.0, weight_outlier=0.05, threshold=3):
    z_scores = np.abs((X - X.mean()) / X.std())
    outlier_mask = (z_scores > threshold).any(axis=1)
    weights = np.where(outlier_mask, weight_outlier, weight_normal)
    return weights

# ------------ 模型训练 ------------
def train_model_with_tuning(X_train_scaled, y_train, X_test_scaled, y_test, weights, use_tuning=True):
    if use_tuning:
        param_dist = {
            'learning_rate': list(np.arange(0.01, 0.11, 0.01)),
            'max_depth': list(range(4, 11)),
            'min_child_weight': list(range(1, 6)),
            'gamma': list(np.arange(0, 0.6, 0.1)),
            'subsample': list(np.arange(0.5, 1.01, 0.05)),
            'colsample_bytree': list(np.arange(0.5, 1.01, 0.05)),
            'reg_alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5],
            'reg_lambda': list(np.arange(1.0, 1.6, 0.1))
        }
        xgb_reg = XGBRegressor(objective='reg:squarederror', eval_metric='rmse',
                               n_estimators=NUM_BOOST_ROUND, seed=RANDOM_STATE)
        tscv = TimeSeriesSplit(n_splits=3)
        extra_fit_params = {
            'eval_set': [(X_train_scaled, y_train), (X_test_scaled, y_test)],
            'early_stopping_rounds': 20,
            'verbose': 200  # 每200轮输出一次验证指标
        }
        if SEARCH_MODE == "grid":
            search = GridSearchCV(
                estimator=xgb_reg,
                param_grid=param_dist,
                scoring='neg_mean_squared_error',
                cv=tscv,
                verbose=1,
                n_jobs=-1
            )
        elif SEARCH_MODE == "bayesian":
            search = BayesSearchCV(
                estimator=xgb_reg,
                search_spaces=param_dist,
                n_iter=50,
                scoring='neg_mean_squared_error',
                cv=tscv,
                random_state=RANDOM_STATE,
                verbose=1,
                n_jobs=-1
            )
        else:
            search = RandomizedSearchCV(
                estimator=xgb_reg,
                param_distributions=param_dist,
                n_iter=50,
                scoring='neg_mean_squared_error',
                cv=tscv,
                random_state=RANDOM_STATE,
                verbose=1,
                n_jobs=-1
            )
        search.fit(X_train_scaled, y_train, sample_weight=weights)
        best_model = search.best_estimator_
        print("调优后的最佳参数:", search.best_params_)
        best_model.fit(X_train_scaled, y_train,
                       eval_set=[(X_test_scaled, y_test)],
                       verbose=200)
    else:
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train, weight=weights)
        dtest = xgb.DMatrix(X_test_scaled, label=y_test)
        best_model = xgb.train(DEFAULT_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND,
                               evals=[(dtrain, 'Train'), (dtest, 'Test')],
                               verbose_eval=False)
    return best_model

# ------------ 模型评价与预测 ------------
def evaluate_and_predict(model, scaler, X_train, y_train, X_test, y_test, X_future, use_tuning=True):
    X_train_trans = scaler.transform(X_train)
    X_test_trans = scaler.transform(X_test)
    X_future_trans = scaler.transform(X_future)
    
    if isinstance(model, xgb.Booster):
        y_train_pred = model.predict(xgb.DMatrix(X_train_trans))
        y_test_pred = model.predict(xgb.DMatrix(X_test_trans))
        y_future_pred = model.predict(xgb.DMatrix(X_future_trans))
    else:
        y_train_pred = model.predict(X_train_trans)
        y_test_pred = model.predict(X_test_trans)
        y_future_pred = model.predict(X_future_trans)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    if len(y_test) < 2:
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = None
        print("测试集样本不足，R² 无法计算")
    else:
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
    print(f"Train MSE: {train_mse}, Train R²: {train_r2}")
    print(f"Test MSE: {test_mse}, Test R²: {test_r2}")
    
    return y_train_pred, y_test_pred, y_future_pred

# ------------ 结果后处理与保存 ------------
def merge_and_save_results(train_data, test_data, future_data, y_test_pred, y_future_pred):
    test_data = test_data.copy()
    future_data = future_data.copy()
    test_data['预测值'] = y_test_pred
    if '完整数据_预测值' in future_data.columns:
        future_data['预测值'] = future_data['完整数据_预测值']
    else:
        future_data['预测值'] = y_future_pred

    train_data_2023 = train_data[train_data['Date'].dt.year >= 2023][['Date', TARGET_COL]]
    test_actual = test_data[['Date', TARGET_COL]]
    historical_actual = pd.concat([train_data_2023, test_actual])
    historical_actual.columns = ['Date', '实际值']
    
    future_pred = future_data[future_data['Date'] >= '2022-08-01'][['Date', '预测值']].copy()
    future_pred.rename(columns={'预测值': TARGET_COL}, inplace=True)
    last_actual_value = float(historical_actual.iloc[-1]['实际值'])
    future_pred.iloc[0, future_pred.columns.get_loc(TARGET_COL)] = np.float32(last_actual_value)

    merged_df = pd.merge(historical_actual, future_pred, on='Date', how='outer')
    merged_df = merged_df.sort_values('Date', ascending=False)
    merged_df['Date'] = merged_df['Date'].dt.strftime('%Y/%m/%d')
    merged_df.to_excel(OUTPUT_DAILY, index=False, float_format='%.2f')
    
    actual_values = pd.concat([
        train_data[train_data['Date'].dt.year >= 2023][['Date', TARGET_COL]],
        test_actual
    ])
    actual_values.columns = ['Date', '实际值']
    predictions = pd.concat([
        test_data[['Date', '预测值']],
        future_pred.rename(columns={TARGET_COL: '预测值'})
    ], ignore_index=True)
    monthly_df = pd.merge(actual_values, predictions, on='Date', how='outer')
    monthly_df['Date'] = pd.to_datetime(monthly_df['Date'])
    monthly_df.set_index('Date', inplace=True)
    monthly_df = monthly_df.resample('ME').mean()
    monthly_df.reset_index(inplace=True)
    monthly_df = monthly_df.sort_values('Date', ascending=False)
    monthly_df['Date'] = monthly_df['Date'].dt.strftime('%Y/%m/%d')
    monthly_df.to_excel(OUTPUT_MONTHLY, index=False, float_format='%.2f')
    
    return merged_df

def update_excel(merged_df):
    success = update_excel_data(merged_df, UPDATE_FILE_PATH, UPDATE_SHEET_NAME, UPDATE_IDENTIFIER)
    if success:
        print("数据已成功更新到Excel文件")
    else:
        print("数据更新失败，请检查错误信息")

def adjust_full_predictions(y_test, future_data):
    gap = y_test.iloc[-1] - future_data['完整数据_预测值'].iloc[0]
    future_data['完整数据_预测值'] = future_data['完整数据_预测值'] + gap
    print(future_data['完整数据_预测值'])
    return future_data

# ------------ 最终预测结果可视化 ------------
def plot_final_predictions(train_data, y_train, y_train_pred,
                           test_data, y_test, y_test_pred,
                           future_data, last_day):
    plt.figure(figsize=(15, 6))
    plt.plot(train_data['Date'], y_train, label='Train True', color='blue')
    plt.plot(train_data['Date'], y_train_pred, label='Train Predicted', color='green')
    plt.plot(test_data['Date'], y_test, label='Test True', color='blue', alpha=0.7)
    plt.plot(test_data['Date'], y_test_pred, label='Test Predicted', color='red')
    plt.plot(future_data['Date'], future_data['预测值'], label='Future Prediction', color='purple')
    if '完整数据_预测值' in future_data.columns:
        plt.plot(future_data['Date'], future_data['完整数据_预测值'], label='Full Model Future Prediction', color='black')
    plt.axvline(x=test_data['Date'].iloc[0], color='black', linestyle='--', label='Train/Test Split')
    plt.axvline(x=last_day, color='gray', linestyle='--', label='Future Split')
    plt.title('Prediction Visualization')
    plt.xlabel('Date')
    plt.ylabel('Target Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# ------------ 全数据训练及未来预测 ------------
def train_full_model_and_predict(X_train, y_train, X_test, y_test, X_future):
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])
    scaler_full = StandardScaler().fit(X_full)
    X_full_scaled = scaler_full.transform(X_full)
    X_future_scaled = scaler_full.transform(X_future)
    
    params = None
    if USE_HYPERPARAM_TUNING:
        params = None
    if params is None:
        params = DEFAULT_PARAMS
    full_model = XGBRegressor(**params, n_estimators=NUM_BOOST_ROUND)
    full_model.fit(X_full_scaled, y_full)
    y_future_full_pred = full_model.predict(X_future_scaled)
    return full_model, y_future_full_pred, scaler_full

# ------------ 主函数 ------------
def main():
    df_daily, last_day = load_and_preprocess_data(FILE_PATH)
    X_train, y_train, X_test, y_test, X_future, train_data, test_data, future_data = split_and_build_features(df_daily, last_day)
    scaler, X_train_scaled, X_test_scaled, X_future_scaled = scale_and_weight_features(X_train, X_test, X_future)
    weights = detect_outliers_weights(X_train, weight_normal=1.0, weight_outlier=0.05, threshold=3)
    
    model = train_model_with_tuning(X_train_scaled, y_train, X_test_scaled, y_test, weights,
                                    use_tuning=USE_HYPERPARAM_TUNING)
    y_train_pred, y_test_pred, y_future_pred = evaluate_and_predict(model, scaler, X_train, y_train, X_test, y_test, X_future,
                                                                    use_tuning=USE_HYPERPARAM_TUNING)
    
    test_data = test_data.copy()
    test_data['预测值'] = y_test_pred
    future_data = future_data.copy()
    future_data['预测值'] = y_future_pred
    
    full_model, y_future_full_pred, scaler_full = train_full_model_and_predict(X_train, y_train, X_test, y_test, X_future)
    future_data['完整数据_预测值'] = y_future_full_pred
    

    if ADJUST_FULL_PREDICTIONS:
        future_data = adjust_full_predictions(y_test, future_data)
    
    merged_df = merge_and_save_results(train_data, test_data, future_data, y_test_pred, y_future_full_pred)
    update_excel(merged_df)
    
    if SHOW_PLOTS:
        plot_final_predictions(train_data, y_train, y_train_pred,
                               test_data, y_test, y_test_pred,
                               future_data, last_day)
    
    print("全数据模型对未来数据的预测结果：", y_future_full_pred)

if __name__ == '__main__':
    main()
