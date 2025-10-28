import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit

from Dtool import fill_missing_values, reverse_column
from api import fetch_data_by_indicators


# 使用示例
indicators = ["RBWTICKMc1", "C2406121350446455",'USGGBE02 Index', "Cinjcjc4 index",'injcjc4 index','C2201059138_241106232710','C2406036178','C22411071623523660','C2312081670','REFOC-T-EIA_241114135248','C2304065621_241024124344','REFOC-T-EIA_241114135248','C22503031424010431']
df = fetch_data_by_indicators(indicators)
df = fetch_data_by_indicators(indicators, "data_input/RBOB.xlsx")


# ------------ 全局配置参数 ------------
FILE_PATH = 'data_input/RBOB.xlsx'

NUM_BOOST_ROUND = 1000
RANDOM_STATE = 42
USE_HYPERPARAM_TUNING = False    # 若 False 则直接使用 xgb.train
TARGET_COL = '美国RBOB汽油裂解'
TEST_PERIOD = 20
SEARCH_MODE = 'random'           # 可选 'grid' / 'bayesian' / 'random'
SHOW_PLOTS = False
ADJUST_FULL_PREDICTIONS = True

DEFAULT_PARAMS = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.1,
    'max_depth': 8,
    'min_child_weight': 3,
    'gamma': 2,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'eval_metric': 'rmse',
    'seed': 42,
    'reg_alpha': 0.45,
    'reg_lambda': 1.29,
}

# —— 因子预处理相关配置 —— 
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
    ('美国EIA成品油总库存(预测/供应需求3年季节性)', 15,
     '美国EIA成品油总库存(预测/供应需求3年季节性)_提前15天'),
    ('美国成品车用汽油炼厂与调和装置净产量/4WMA(预测/上年季节性)超季节性/5年',
     30,
     '美国成品车用汽油炼厂与调和装置净产量/4WMA(预测/上年季节性)超季节性/5年_提前30天'),
    ('美国炼厂CDU装置检修量（新）', 30, '美国炼厂CDU装置检修量（新）_提前30天'),
    ('美国炼厂可用产能（路透）(预测)', 100,
     '美国炼厂可用产能（路透）(预测)_提前100天')
]

REVERSE_CONFIG = [
    ('美国首次申领失业金人数/4WMA',
     '美国首次申领失业金人数/4WMA_逆序'),
    ('美国首次申领失业金人数/4WMA_提前100天',
     '美国首次申领失业金人数/4WMA_提前100天_逆序'),
    ('美国首次申领失业金人数/4WMA_提前112天',
     '美国首次申领失业金人数/4WMA_提前112天_逆序'),
    ('美国EIA成品油总库存(预测/供应需求3年季节性)',
     '美国EIA成品油总库存(预测/供应需求3年季节性)_逆序'),
    ('美国EIA成品油总库存(预测/供应需求3年季节性)_提前15天',
     '美国EIA成品油总库存(预测/供应需求3年季节性)_提前15天_逆序'),
    ('美国炼厂可用产能（路透）(预测)_提前100天',
     '美国炼厂可用产能（路透）(预测)_逆序'),
    ('美国汽油调和组分RBOB库存(预测/线性外推)超季节性/3年',
     '美国汽油调和组分RBOB库存(预测/线性外推)超季节性/3年_逆序')
]

SPECIAL_REVERSE = {
    '美国汽油调和组分RBOB库存(预测/线性外推)超季节性/3年_逆序_2022-01-01': {
        'base_column': '美国汽油调和组分RBOB库存(预测/线性外推)超季节性/3年_逆序',
        'condition_date': pd.Timestamp('2022-01-01')
    }
}


# ------------ 数据加载与预处理 ------------
def load_and_preprocess_data(file_path):
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    print(df)
    df.rename(columns={'DataTime': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.set_index('Date', inplace=True)

    full_range = pd.date_range(df.index.min(),df.index.max(),freq='D')
    df_daily = df.reindex(full_range).reset_index()
    df_daily.rename(columns={'index': 'Date'}, inplace=True)
    df_daily = fill_missing_values(df_daily,FILL_METHODS,return_only_filled=False)
    for col, days, new_col in SHIFT_CONFIG:
        df_daily[new_col] = df_daily[col].shift(days)

    last_idx = df_daily[TARGET_COL].last_valid_index()
    last_day = df_daily.loc[last_idx, 'Date']

    df_daily = df_daily[(df_daily['Date'] >= '2009-08-01') &(df_daily['Date'] <= last_day +pd.Timedelta(days=30))]
    df_daily = df_daily[df_daily['Date'].dt.weekday < 5]

    for base, new in REVERSE_CONFIG:
        df_daily[new] = reverse_column(df_daily, base)
    for col, cfg in SPECIAL_REVERSE.items():
        df_daily[col] = np.where(df_daily['Date'] >= cfg['condition_date'],df_daily[cfg['base_column']],np.nan)

    df_daily = df_daily[(df_daily['Date'] > last_day)|df_daily[TARGET_COL].notna()]

    return df_daily, last_day


# ------------ 划分与特征构建 ------------
def split_and_build_features(df_daily, last_day):
    train = df_daily[df_daily['Date'] <= last_day].copy()
    test = train.tail(TEST_PERIOD).copy()
    train = train.iloc[:-TEST_PERIOD].copy()
    future = df_daily[df_daily['Date'] > last_day].copy()

    feature_columns = [
        '美湾单位辛烷值价格(预测/季节性)',
        '美国炼厂CDU装置检修量（新）_提前30天',
        '美国EIA成品油总库存(预测/供应需求3年季节性)_提前15天_逆序',
        '美国首次申领失业金人数/4WMA_提前100天_逆序',
        '美国成品车用汽油倒推产量（预测/汽油库存维持上年季节性）/8WMA',
        '美国成品车用汽油炼厂与调和装置净产量/4WMA(预测/上年季节性)超季节性/5年_提前30天',
        '美国汽油调和组分RBOB库存(预测/线性外推)超季节性/3年_逆序_2022-01-01'
    ]

    X_train = train[feature_columns]
    y_train = train[TARGET_COL]
    X_test = test[feature_columns]
    y_test = test[TARGET_COL]
    X_future = future[feature_columns]

    return X_train, y_train, X_test, y_test, X_future, train, test, future


# ------------ 特征缩放与异常值权重 ------------
def scale_and_weight_features(X_train, X_test, X_future):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    X_fu = scaler.transform(X_future)
    return scaler, X_tr, X_te, X_fu


def detect_outliers_weights(X,weight_normal=1.0,weight_outlier=0.05,threshold=3):
    z = np.abs((X - X.mean()) / X.std())
    mask = (z > threshold).any(axis=1)
    return np.where(mask, weight_outlier, weight_normal)


# ------------ 模型训练 ------------
def train_model_with_tuning(X_tr, y_tr, X_te, y_te, weights, use_tuning):
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
        xgb_reg = XGBRegressor(objective='reg:squarederror',
                               eval_metric='rmse',
                               n_estimators=NUM_BOOST_ROUND,
                               seed=RANDOM_STATE)
        tscv = TimeSeriesSplit(n_splits=3)
        if SEARCH_MODE == 'grid':
            search = GridSearchCV(xgb_reg,
                                  param_grid=param_dist,
                                  scoring='neg_mean_squared_error',
                                  cv=tscv,
                                  verbose=1,
                                  n_jobs=-1)
        elif SEARCH_MODE == 'bayesian':
            search = BayesSearchCV(xgb_reg,
                                  search_spaces=param_dist,
                                  n_iter=50,
                                  scoring='neg_mean_squared_error',
                                  cv=tscv,
                                  random_state=RANDOM_STATE,
                                  verbose=1,
                                  n_jobs=-1)
        else:
            search = RandomizedSearchCV(xgb_reg,
                                        param_distributions=param_dist,
                                        n_iter=50,
                                        scoring='neg_mean_squared_error',
                                        cv=tscv,
                                        random_state=RANDOM_STATE,
                                        verbose=1,
                                        n_jobs=-1)
        search.fit(X_tr, y_tr, sample_weight=weights)
        best_model = search.best_estimator_
        print("调优后的最佳参数:", search.best_params_)
        best_model.fit(X_tr, y_tr,
                       eval_set=[(X_te, y_te)],
                       early_stopping_rounds=20,
                       verbose=200)
    else:
        dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=weights)
        dtest = xgb.DMatrix(X_te, label=y_te)
        best_model = xgb.train(DEFAULT_PARAMS,
                               dtrain,
                               num_boost_round=NUM_BOOST_ROUND,
                               evals=[(dtrain, 'Train'),
                                      (dtest, 'Test')],
                               verbose_eval=False)
    return best_model


# ------------ 评估与预测 ------------
def evaluate_and_predict(model, scaler, X_tr, y_tr, X_te, y_te, X_fu,
                         use_tuning):
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)
    X_fu_s = scaler.transform(X_fu)

    if isinstance(model, xgb.Booster):
        y_tr_pred = model.predict(xgb.DMatrix(X_tr_s))
        y_te_pred = model.predict(xgb.DMatrix(X_te_s))
        y_fu_pred = model.predict(xgb.DMatrix(X_fu_s))
    else:
        y_tr_pred = model.predict(X_tr_s)
        y_te_pred = model.predict(X_te_s)
        y_fu_pred = model.predict(X_fu_s)

    print("Train MSE:", mean_squared_error(y_tr, y_tr_pred),
          "Test MSE:", mean_squared_error(y_te, y_te_pred))
    if len(y_te) >= 2:
        print("Train R2:", r2_score(y_tr, y_tr_pred),
              "Test R2:", r2_score(y_te, y_te_pred))
    else:
        print("Test 样本不足，跳过 R² 计算")

    return y_tr_pred, y_te_pred, y_fu_pred


# ------------ 结果后处理（生成日度 & 月度 DataFrame） ------------
def merge_and_prepare_df(train, test, future, y_te_pred, y_fu_pred):
    # 合并历史与未来预测
    test = test.copy()
    future = future.copy()
    test['预测值'] = y_te_pred
    future['预测值'] = y_fu_pred

    hist_actual = pd.concat([
        train[train['Date'].dt.year >= 2023][['Date', TARGET_COL]],
        test[['Date', TARGET_COL]]
    ])
    hist_actual.columns = ['Date', '实际值']

    future_pred = future[future['Date'] >= '2022-08-01'][['Date', '预测值']].rename(columns={'预测值': TARGET_COL}).copy()

    last_val = float(hist_actual.iloc[-1]['实际值'])
    future_pred.iloc[0, 1] = last_val

    merged = pd.merge(hist_actual, future_pred,on='Date', how='outer').sort_values('Date', ascending=False)
    daily_df = merged.copy()

    # 月度重采样
    monthly_df = daily_df.copy()
    monthly_df['Date'] = pd.to_datetime(monthly_df['Date'])
    monthly_df.set_index('Date', inplace=True)
    monthly_df = monthly_df.resample('ME').mean().reset_index()

    # 方向准确率：仅在实际值和预测值都非空时计算
    pred_dir = np.sign(monthly_df[TARGET_COL].diff())
    true_dir = np.sign(monthly_df['实际值'].diff())
    valid = monthly_df[TARGET_COL].notna() & monthly_df['实际值'].notna()
    monthly_df['方向准确率'] = np.where(valid & (pred_dir == true_dir), '正确',
                                   np.where(valid & (pred_dir != true_dir), '错误', np.nan))
    # 绝对偏差
    monthly_df['绝对偏差'] = (monthly_df[TARGET_COL] - monthly_df['实际值']).abs()
    # 降序 & 打印
    monthly_df = monthly_df.sort_values('Date', ascending=False).reset_index(drop=True)

    return daily_df, monthly_df



def generate_and_fill_excel(
    daily_df,
    monthly_df,
    target_name,        # 写入的“预测标的”显示名
    classification,     # 列表页-分类
    model_framework,    # 列表页-模型框架
    creator,            # 列表页-创建人
    pred_date,          # 列表页-预测日期
    frequency,          # 列表页-预测频度
    output_path='update.xlsx'
):
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        workbook = writer.book

        # —— 计算三个汇总值 —— 
        # 1) 测试值：最新月度的预测值
        test_value = monthly_df[TARGET_COL].iloc[0]
        # 2) 方向准确率：正确数 / 有效数
        total = monthly_df['方向准确率'].notna().sum()
        correct = (monthly_df['方向准确率'] == '正确').sum()
        direction_accuracy = f"{correct/total:.2%}" if total > 0 else ""
        # 3) 平均绝对偏差
        absolute_deviation = monthly_df['绝对偏差'].mean()

        # ========= 列表页 =========
        ws_list = workbook.add_worksheet('列表页')
        writer.sheets['列表页'] = ws_list

        headers = ['预测标的','分类','模型框架','创建人','预测日期','测试值','预测频度','方向准确率','绝对偏差']

        ws_list.write_row(0, 0, headers)
        ws_list.write_row(1, 0, [
            target_name,
            classification,
            model_framework,
            creator,
            pred_date,
            test_value,
            frequency,
            direction_accuracy,
            absolute_deviation
        ])

        # ========= 详情页 =========
        detail_df = monthly_df[['Date', '实际值', TARGET_COL, '方向准确率', '绝对偏差']].copy()
        detail_df.columns = ['指标日期','实际值','预测值','方向','偏差率']

        detail_df.to_excel(writer,sheet_name='详情页',index=False,header=False,startrow=2)

        ws_detail = writer.sheets['详情页']
        ws_detail.write(0, 0, target_name)
        ws_detail.write_row(1, 0, ['指标日期','实际值','预测值','方向','偏差率'])

        # ========= 日度数据表 =========
        daily_out = daily_df[['Date', '实际值', TARGET_COL]].copy()
        daily_out.columns = ['指标日期','实际值','预测值']

        daily_out.to_excel(writer,sheet_name='日度数据表',index=False,header=False,startrow=2)
        
        ws_daily = writer.sheets['日度数据表']
        ws_daily.write(0, 0, target_name)
        ws_daily.write_row(1, 0, ['指标日期','实际值','预测值'])

    print(f"已生成并填充 {output_path}")


# ------------ 全量训练与预测 ------------
def train_full_model_and_predict(X_train, y_train, X_test, y_test, X_future):
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])
    scaler_all = StandardScaler().fit(X_all)
    X_all_s = scaler_all.transform(X_all)
    X_fu_s = scaler_all.transform(X_future)

    model = XGBRegressor(**DEFAULT_PARAMS, n_estimators=NUM_BOOST_ROUND)
    model.fit(X_all_s, y_all)
    y_fu_full = model.predict(X_fu_s)

    return model, y_fu_full, scaler_all


# ------------ 可视化 ------------
def plot_final_predictions(train, y_tr, y_tr_pred, test, y_te, y_te_pred,
                           future, last_day):
    plt.figure(figsize=(15, 6))
    plt.plot(train['Date'], y_tr, label='Train True')
    plt.plot(train['Date'], y_tr_pred, label='Train Pred')
    plt.plot(test['Date'], y_te, label='Test True', alpha=0.7)
    plt.plot(test['Date'], y_te_pred, label='Test Pred')
    plt.plot(future['Date'], future['预测值'], label='Future Pred')
    plt.axvline(test['Date'].iloc[0], color='gray', linestyle='--')
    plt.axvline(last_day, color='black', linestyle='--')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel(TARGET_COL)
    plt.title('Prediction Visualization')
    plt.grid(True)
    plt.show()


# ------------ 主函数 ------------
def main():
    df_daily, last_day = load_and_preprocess_data(FILE_PATH)

    X_tr, y_tr, X_te, y_te, X_fu, train, test, future = split_and_build_features(df_daily, last_day)

    scaler, X_tr_s, X_te_s, X_fu_s = scale_and_weight_features(X_tr, X_te, X_fu)

    weights = detect_outliers_weights(X_tr_s)

    model = train_model_with_tuning(X_tr_s, y_tr, X_te_s, y_te, weights,USE_HYPERPARAM_TUNING)

    y_tr_pred, y_te_pred, y_fu_pred = evaluate_and_predict(model, scaler, X_tr, y_tr, X_te, y_te, X_fu,USE_HYPERPARAM_TUNING)

    daily_df, monthly_df = merge_and_prepare_df(train, test, future,y_te_pred, y_fu_pred)

    print(monthly_df)
    print(daily_df)

    generate_and_fill_excel(
        daily_df,
        monthly_df,
        target_name='美国RBOB汽油裂解',        
        classification='原油',
        model_framework='XGBoost',
        creator='张立舟',
        pred_date='2024/11/11',             
        frequency='月度',
        output_path='update.xlsx'    
    )
    
    full_model, y_fu_full, scaler_full = train_full_model_and_predict(X_tr, y_tr, X_te, y_te, X_fu)

    if ADJUST_FULL_PREDICTIONS:
        offset = y_te.iloc[-1] - y_fu_full[0]
        y_fu_full += offset

    if SHOW_PLOTS:
        plot_final_predictions(
            train, y_tr, y_tr_pred, test, y_te, y_te_pred,
            future.assign(预测值=y_fu_full), last_day)

    return daily_df, monthly_df

if __name__ == '__main__':
    daily_df, monthly_df = main()
