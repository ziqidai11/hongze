# -*- coding: utf-8 -*-
"""
模块名称: model_pipeline
模块描述: 一个通用的XGBoost时间序列建模流水线，封装了数据加载、预处理、因子处理、模型训练（部分训练和全量训练）、预测、可视化以及结果导出
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os
import warnings

# 优先调用 Dtool 中的函数，如果可用（请确保 Dtool.py 在同一目录下）
from Dtool import fill_missing_values as dt_fill_missing_values, reverse_column as dt_reverse_column, plot_comparison_multiple

# ---------------- 辅助函数 ----------------

def fill_missing_values(df, fill_methods, return_only_filled=False):
    """
    填充缺失值，支持针对不同列采用不同的填充方法。
    参数:
        df: 输入的 DataFrame
        fill_methods: dict，键为列名称，值为填充方法，例如 'interpolate'
        return_only_filled: 若为 True，则只返回填充的那几列，否则返回整个 DataFrame
    优先调用 Dtool 中的方法
    """
    if dt_fill_missing_values is not None:
        return dt_fill_missing_values(df, fill_methods, return_only_filled)
    df_filled = df.copy()
    for col, method in fill_methods.items():
        if method == 'interpolate':
            df_filled[col] = df_filled[col].interpolate(method='linear')
        # 可扩展其他方法，如 'ffill', 'bfill', 'mean' 等
    if return_only_filled:
        return df_filled[list(fill_methods.keys())]
    return df_filled

def reverse_column(df, column):
    """
    将指定列数据逆序排列，生成新序列。
    优先调用 Dtool 中的实现
    """
    if dt_reverse_column is not None:
        return dt_reverse_column(df, column)
    return df[column][::-1]

def slice_factor_by_date(df, factor, slice_date, new_column_suffix="_slice"):
    """
    根据指定日期截取因子数据——例如在给出的美国10债Non-trend.ipynb中，仅保留指定日期之后的数据，其余赋值NaN。
    参数:
        df: 输入 DataFrame（日期信息应在索引中或包含 'Date' 列）
        factor: 待截取的因子列名称
        slice_date: 截取的起始日期（pd.Timestamp 或可转换为Timestamp的字符串）
        new_column_suffix: 新生成列的后缀，默认 "_slice"
    返回:
        修改后的 DataFrame，新列名称为 factor + new_column_suffix
    """
    slice_date = pd.to_datetime(slice_date)
    new_col = factor + new_column_suffix
    if df.index.dtype == 'datetime64[ns]':
        df[new_col] = np.where(df.index >= slice_date, df[factor], np.nan)
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df[new_col] = np.where(df['Date'] >= slice_date, df[factor], np.nan)
    else:
        raise ValueError("DataFrame中未找到日期索引或'Date'列")
    return df

def adjust_future_predictions(y_test, future_data, model_all_pred_col='预测值_全量'):
    """
    调整未来预测值，使得测试集最后的真实值与全量预测之间平滑衔接。
    参数:
        y_test: 测试集真实值（Series）
        future_data: 包含全量预测列的 DataFrame，必须包含 model_all_pred_col 列
    返回:
        修改后的 future_data，增加 '预测值_全量_移动' 列
    """
    if len(y_test) == 0 or future_data.empty:
        raise ValueError("y_test 或 future_data 为空")
    gap = y_test.iloc[-1] - future_data[model_all_pred_col].iloc[0]
    future_data['预测值_全量_移动'] = future_data[model_all_pred_col] + gap
    return future_data

def plot_training_curve(evals_result):
    """
    绘制模型训练过程中误差（例如RMSE）随迭代次数的变化曲线。
    参数:
        evals_result: XGBoost训练后返回的评估结果字典
    """
    if evals_result is None:
        print("没有评估结果可绘图。")
        return
    train_rmse = evals_result.get('train', {}).get('rmse', [])
    if not train_rmse:
        print("评估结果中未找到 'rmse' 数据")
        return
    plt.figure(figsize=(10,6))
    plt.plot(train_rmse, label='Train RMSE')
    plt.xlabel('Boost round')
    plt.ylabel('RMSE')
    plt.title('训练过程中的RMSE曲线')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(train_data, y_train, train_pred, test_data, y_test, test_pred, future_data, future_pred, date_column='Date'):
    """
    绘制训练集、测试集和未来数据的预测结果对比图。
    参数:
        train_data, test_data, future_data: 含日期信息的 DataFrame
        y_train, y_test: 真实值（Series）
        train_pred, test_pred, future_pred: 预测值（数组或list）
        date_column: 日期列名称；如不存在，则尝试使用索引
    """
    plt.figure(figsize=(15,6))
    # 训练集
    if date_column in train_data.columns:
        dates_train = train_data[date_column]
    else:
        dates_train = train_data.index
    plt.plot(dates_train, y_train, label='训练集-真实值', color='blue')
    plt.plot(dates_train, train_pred, label='训练集-预测', color='green')
    # 测试集
    if date_column in test_data.columns:
        dates_test = test_data[date_column]
    else:
        dates_test = test_data.index
    plt.plot(dates_test, y_test, label='测试集-真实值', color='blue', alpha=0.7)
    plt.plot(dates_test, test_pred, label='测试集-预测', color='purple')
    # 未来预测：此处用红色表示部分数据训练的模型预测，黑色表示全量数据训练预测（若存在）
    if date_column in future_data.columns:
        dates_future = future_data[date_column]
    else:
        dates_future = future_data.index
    plt.plot(dates_future, future_pred, label='未来预测（部分训练）', color='red')
    if '预测值_全量' in future_data.columns:
        plt.plot(dates_future, future_data['预测值_全量'], label='未来预测（全量训练）', color='black')
    plt.xlabel('日期')
    plt.ylabel('目标值')
    plt.title('预测结果对比图')
    plt.legend()
    plt.grid(True)
    plt.show()

def export_results(merged_df, output_path, output_format='excel', float_format='%.4f'):
    """
    将结果DataFrame导出为Excel或CSV文件。
    参数:
        merged_df: 待导出的DataFrame
        output_path: 文件路径
        output_format: 'excel' 或 'csv'
        float_format: 数值格式化格式
    """
    if output_format == 'excel':
        merged_df.to_excel(output_path, index=False, float_format=float_format)
    elif output_format == 'csv':
        merged_df.to_csv(output_path, index=False, float_format=float_format)
    else:
        raise ValueError("输出格式仅支持 'excel' 或 'csv'")

def export_forecast_results(train_data, test_data, future_data, y_train, y_test, y_train_pred, y_test_pred, future_pred_all,
                            daily_output_path, monthly_output_path, date_column='Date', target_column='真实值'):
    """
    参照USDCNY即期汇率.ipynb的导出逻辑，导出日度和月度预测结果。
    参数:
        train_data, test_data: 包含日期的DataFrame（训练和测试数据）
        y_train, y_test: 真实值（Series）
        y_train_pred, y_test_pred: 训练集和测试集预测值
        future_pred_all: 全量数据训练预测的未来值（数组）
        daily_output_path: 日度数据Excel导出路径
        monthly_output_path: 月度数据Excel导出路径
        date_column: 日期列名称
        target_column: 真实值列名称（默认 '真实值'）
    """
    # 日度数据组装
    historical_actual_daily = pd.DataFrame({
        'Date': pd.concat([train_data[date_column], test_data[date_column]]),
        target_column: pd.concat([y_train, y_test])
    })
    historical_actual_daily = historical_actual_daily[historical_actual_daily['Date'].dt.year >= 2023]
    future_pred_daily = pd.DataFrame({
        'Date': future_data[date_column],
        '预测值': future_pred_all
    })
    merged_df_daily = pd.merge(historical_actual_daily, future_pred_daily, on='Date', how='outer')
    merged_df_daily = merged_df_daily.sort_values('Date', ascending=False)
    merged_df_daily['Date'] = merged_df_daily['Date'].dt.strftime('%Y/%m/%d')
    export_results(merged_df_daily, daily_output_path, output_format='excel', float_format='%.4f')

    # 月度数据组装
    historical_actual_monthly = pd.DataFrame({
        'Date': pd.concat([train_data[date_column], test_data[date_column]]),
        target_column: pd.concat([y_train, y_test])
    })
    historical_actual_monthly = historical_actual_monthly[historical_actual_monthly['Date'].dt.year >= 2023]
    train_pred_2024 = pd.DataFrame({
        'Date': train_data[train_data[date_column].dt.year >= 2024][date_column],
        '预测值': y_train_pred[train_data[train_data[date_column].dt.year >= 2024].index]
    })
    test_pred_2024 = test_data[[date_column]].copy()
    test_pred_2024['预测值'] = y_test_pred
    future_pred_2024 = pd.DataFrame({
        'Date': future_data[date_column],
        '预测值': future_pred_all
    })
    future_pred_monthly = pd.concat([train_pred_2024, test_pred_2024, future_pred_2024], ignore_index=True).sort_values('Date')
    merged_df_monthly = pd.merge(historical_actual_monthly, future_pred_monthly, on='Date', how='outer')
    merged_df_monthly.set_index('Date', inplace=True)
    merged_df_monthly = merged_df_monthly.resample('ME').mean()
    merged_df_monthly.reset_index(inplace=True)
    merged_df_monthly = merged_df_monthly.sort_values('Date', ascending=False)
    merged_df_monthly['Date'] = merged_df_monthly['Date'].dt.strftime('%Y/%m/%d')
    export_results(merged_df_monthly, monthly_output_path, output_format='excel', float_format='%.4f')

# ---------------- 数据加载、预处理及其他通用函数 ----------------

def load_data(file_path, sheet_name='Sheet1', rename_cols=None, date_column='Date', set_index=True):
    """
    从Excel文件中加载数据，并对日期进行格式转换。
    参数:
        file_path: 文件路径
        sheet_name: Excel的Sheet名称
        rename_cols: 列重命名字典，例如 {'DataTime': 'Date'}
        date_column: 日期列名称
        set_index: 是否将日期列设置为索引（默认为True）
    返回:
        加载后的 DataFrame
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    if rename_cols:
        df = df.rename(columns=rename_cols)
    df[date_column] = pd.to_datetime(df[date_column])
    if set_index:
        df = df.set_index(date_column)
    return df

def preprocess_data(df, fill_methods=None, shift_features=None, reverse_features=None,
                    date_filter=None, sort_index=True):
    """
    对数据进行预处理：
      1. 缺失值填充
      2. 生成提前特征（shift操作），新列名格式：原列名_提前{n}天
      3. 对指定列进行逆序处理，生成新列名：原列名_逆序
      4. 根据日期筛选数据（依据索引或 'Date' 列）
      5. 按日期排序
    参数:
        df: 输入的 DataFrame（建议日期列已转换为 datetime 格式）
        fill_methods: dict，缺失值填充设置，如 {'col1': 'interpolate'}
        shift_features: dict，提前特征参数，如 {'原始列': 提前天数}
        reverse_features: list，逆序处理的列名列表
        date_filter: pd.Timestamp，筛选数据的起始日期
        sort_index: 是否排序，默认为True
    返回:
        预处理后的 DataFrame
    """
    df_processed = df.copy()
    # 1. 缺失值填充
    if fill_methods:
        df_processed = fill_missing_values(df_processed, fill_methods, return_only_filled=False)
    # 2. 生成提前特征
    if shift_features:
        for col, shift_days in shift_features.items():
            new_col = f"{col}_提前{shift_days}天"
            df_processed[new_col] = df_processed[col].shift(shift_days)
    # 3. 逆序处理
    if reverse_features:
        for col in reverse_features:
            new_col = f"{col}_逆序"
            df_processed[new_col] = reverse_column(df_processed, col)
    # 4. 日期筛选
    if date_filter is not None:
        if df_processed.index.dtype == 'datetime64[ns]':
            df_processed = df_processed[df_processed.index >= date_filter]
        elif 'Date' in df_processed.columns:
            df_processed['Date'] = pd.to_datetime(df_processed['Date'])
            df_processed = df_processed[df_processed['Date'] >= date_filter]
    # 5. 排序
    if sort_index:
        if df_processed.index.dtype == 'datetime64[ns]':
            df_processed = df_processed.sort_index()
        elif 'Date' in df_processed.columns:
            df_processed = df_processed.sort_values('Date')
    return df_processed

def split_data(df, last_day, test_period=10, date_column='Date'):
    """
    将数据分割为训练集、测试集和未来数据。
    参数:
        df: 待分割的 DataFrame（日期应在索引中或由 date_column 指定）
        last_day: pd.Timestamp，训练集最后一条数据的日期
        test_period: 整数，测试集的样本数（默认10）
        date_column: 如未设置索引，则指定日期列名称
    返回:
        train_data, test_data, future_data 三个 DataFrame
    """
    if df.index.dtype == 'datetime64[ns]':
        df_reset = df.reset_index()
    else:
        df_reset = df.copy()
    df_reset[date_column] = pd.to_datetime(df_reset[date_column])
    # 训练集：<= last_day，未来数据：> last_day
    train_data = df_reset[df_reset[date_column] <= last_day].copy()
    future_data = df_reset[df_reset[date_column] > last_day].copy()
    # 从训练集尾部取 test_period 条作为测试集
    if len(train_data) > test_period:
        test_data = train_data.iloc[-test_period:].copy()
        train_data = train_data.iloc[:-test_period].copy()
    else:
        test_data = train_data.copy()
        train_data = pd.DataFrame(columns=train_data.columns)
    return train_data, test_data, future_data

def scale_features(X_train, X_test, X_future, scaler=None):
    """
    对特征数据进行标准化
    参数:
        X_train, X_test, X_future: 输入特征（DataFrame或ndarray）
        scaler: 标准化工具（默认为StandardScaler），若为 None 则新建一个
    返回:
        X_train_scaled, X_test_scaled, X_future_scaled, scaler
    """
    if scaler is None:
        scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_future_scaled = scaler.transform(X_future)
    return X_train_scaled, X_test_scaled, X_future_scaled, scaler

# ---------------- ModelPipeline 封装 ----------------

class ModelPipeline:
    """
    该类封装了整个建模流水线，从数据加载、预处理、因子处理、数据分割、特征标准化、模型训练预测到结果导出。
    """
    def __init__(self, file_path, sheet_name='Sheet1', rename_cols=None, date_column='Date', set_index=True):
        """
        初始化加载器及相关属性。
        """
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.rename_cols = rename_cols
        self.date_column = date_column
        self.set_index = set_index
        self.df = None
        self.df_processed = None
        self.train_data = None
        self.test_data = None
        self.future_data = None
        self.scaler = None
        self.model = None
        self.model_all = None  # 全量数据训练的模型
        self.evals_result = None
        self.evals_result_all = None
    
    def load_data(self):
        """
        加载数据。
        """
        self.df = load_data(self.file_path, sheet_name=self.sheet_name, rename_cols=self.rename_cols,
                            date_column=self.date_column, set_index=self.set_index)
        return self.df
    
    def preprocess(self, fill_methods=None, shift_features=None, reverse_features=None, date_filter=None, sort_index=True):
        """
        对数据进行预处理（缺失值填充、提前特征、逆序处理等）。
        """
        if self.df is None:
            raise ValueError("请先调用 load_data() 加载数据")
        self.df_processed = preprocess_data(self.df, fill_methods=fill_methods, shift_features=shift_features,
                                            reverse_features=reverse_features, date_filter=date_filter, sort_index=sort_index)
        return self.df_processed
    
    def factor_slice(self, factor, slice_date, new_column_suffix="_slice"):
        """
        对指定因子进行截取处理，仅保留 slice_date 之后的数据，其余赋值为 NaN。
        """
        if self.df_processed is None:
            raise ValueError("请先执行 preprocess() 进行数据预处理")
        self.df_processed = slice_factor_by_date(self.df_processed, factor, slice_date, new_column_suffix)
        return self.df_processed
    
    def split(self, last_day, test_period=10):
        """
        按照指定的 last_day 将数据分割为训练集、测试集和未来数据。
        """
        self.train_data, self.test_data, self.future_data = split_data(self.df_processed, last_day, test_period, date_column=self.date_column)
        return self.train_data, self.test_data, self.future_data
    
    def prepare_and_scale(self, feature_columns, target_column):
        """
        提取特征和目标值，并进行标准化。
        """
        if self.train_data is None or self.test_data is None or self.future_data is None:
            raise ValueError("请先调用 split() 分割数据")
        X_train = self.train_data[feature_columns].copy()
        X_test = self.test_data[feature_columns].copy()
        X_future = self.future_data[feature_columns].copy()
        y_train = self.train_data[target_column].copy()
        y_test = self.test_data[target_column].copy()
        X_train_scaled, X_test_scaled, X_future_scaled, self.scaler = scale_features(X_train, X_test, X_future, scaler=self.scaler)
        return X_train_scaled, y_train, X_test_scaled, y_test, X_future_scaled
    
    def train_model(self, X_train, y_train, X_test, y_test, params=None, num_boost_round=5000,
                    early_stopping_rounds=100, verbose_eval=100, sample_weight_method="huber"):
        """
        训练 XGBoost 模型，分别训练两个模型：
          1. model：仅使用训练集（部分数据）训练
          2. model_all：使用训练集和测试集全量数据训练，用于后续给出更完整的未来预测（如图中黑线预测）
        参数:
            params: XGBoost 参数字典，若为 None 则采用默认参数
            sample_weight_method: 样本权重方法（此处示例不详细实现）
        返回:
            model, evals_result, 训练集预测, 测试集预测
        """
        # 构造 DMatrix 数据结构
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'learning_rate': 0.01,
                'max_depth': 6,
                'eval_metric': 'rmse',
                'seed': 42
            }
        evals = [(dtrain, 'train'), (dtest, 'eval')]
        self.model = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=evals,
                               early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval)
        self.evals_result = self.model.evals_result()
        train_pred = self.model.predict(dtrain)
        test_pred = self.model.predict(dtest)
        
        # 训练全量数据模型
        X_all = np.concatenate([X_train, X_test])
        y_all = np.concatenate([y_train, y_test])
        dall = xgb.DMatrix(X_all, label=y_all)
        self.model_all = xgb.train(params, dall, num_boost_round=num_boost_round, evals=[(dall, 'all')],
                                   verbose_eval=verbose_eval)
        self.evals_result_all = self.model_all.evals_result()
        
        return self.model, self.evals_result, train_pred, test_pred
    
    def predict_future(self, X_future_scaled):
        """
        根据训练好的模型对未来数据进行预测，返回使用部分数据训练得到的预测和全量数据训练得到的预测。
        同时将预测结果保存到 future_data 中。
        """
        if self.model is None or self.model_all is None:
            raise ValueError("模型未训练，请先调用 train_model()")
        dfuture = xgb.DMatrix(X_future_scaled)
        future_pred = self.model.predict(dfuture)
        future_pred_all = self.model_all.predict(dfuture)
        if self.date_column in self.future_data.columns:
            self.future_data['预测值'] = future_pred
            self.future_data['预测值_全量'] = future_pred_all
        else:
            self.future_data.insert(0, '预测值', future_pred)
            self.future_data.insert(1, '预测值_全量', future_pred_all)
        return future_pred, future_pred_all
    
    def export(self, merged_df, output_path, output_format='excel', float_format='%.4f'):
        """
        导出结果数据。
        """
        export_results(merged_df, output_path, output_format=output_format, float_format=float_format)

# ---------------- 示例代码 ----------------

if __name__ == "__main__":
    # 示例：请根据实际情况修改文件路径及参数
    file_path = "data_input/sample.xlsx"  # 修改为实际数据路径
    pipeline = ModelPipeline(file_path, sheet_name='Sheet1', rename_cols={'DataTime': 'Date'}, date_column='Date')
    
    # 1. 数据加载
    df = pipeline.load_data()
    print("原始数据加载完毕。")
    
    # 2. 数据预处理：缺失值填充、提前特征创建、逆序处理等
    fill_methods = {
        '美国制造业PMI(预测/最新)': 'interpolate',
        '美国经济惊喜指数': 'interpolate',
        'COMEX黄金价格Non-Trend/F0.02': 'interpolate'
    }
    shift_features = {
        '美国制造业PMI(预测/最新)': 20,
        '美国经济惊喜指数': 45,
        'COMEX黄金价格Non-Trend/F0.02': 50
    }
    reverse_features = []  # 如需逆序处理，可指定对应列
    date_filter = pd.Timestamp('2022-11-10')
    df_processed = pipeline.preprocess(fill_methods=fill_methods, shift_features=shift_features,
                                       reverse_features=reverse_features, date_filter=date_filter)
    print("数据预处理完毕。")
    
    # 3. 对部分因子进行截取处理，参考美国10债Non-trend.ipynb（例如只保留2024-03-01之后的数据）
    pipeline.factor_slice('美国首次申领失业金人数/4WMA_提前30天_逆序', '2024-03-01', new_column_suffix='_2024-03-01')
    print("因子截取处理完毕。")
    
    # 可选：调用 plot_comparison_multiple 展示因子对比（依赖 Dtool 中的实现）
    # plot_comparison_multiple(df_processed, main_col='主要因子', compare_cols=['因子1', '因子2'], start_date='2022-01-10', end_date='2022-03-10')
    
    # 4. 数据分割：设定训练集最后一天，分割成训练集、测试集和未来数据
    last_day = pd.Timestamp('2024-01-01')
    train_data, test_data, future_data = pipeline.split(last_day, test_period=10)
    print("数据分割完毕。")
    
    # 5. 特征和目标提取及标准化
    feature_columns = ['美国制造业PMI(预测/最新)_提前20天', '美国经济惊喜指数_提前45天', 'COMEX黄金价格Non-Trend/F0.02_提前50天']
    target_column = '美元指数拟合残差/10年期美国国债收益率'
    X_train_scaled, y_train, X_test_scaled, y_test, X_future_scaled = pipeline.prepare_and_scale(feature_columns, target_column)
    
    # 6. 训练XGBoost模型（同时训练 model 和 model_all），支持传入调参参数
    model, evals_result, train_pred, test_pred = pipeline.train_model(
        X_train_scaled, y_train, X_test_scaled, y_test,
        params=None, num_boost_round=5000, early_stopping_rounds=100, verbose_eval=100, sample_weight_method="huber"
    )
    print("模型训练完毕。")
    
    # 7. 利用训练好的模型预测未来数据
    future_pred, future_pred_all = pipeline.predict_future(X_future_scaled)
    print("未来数据预测完毕。")
    
    # 8. 绘制训练过程和预测结果对比图
    plot_training_curve(evals_result)
    plot_predictions(train_data, y_train, train_pred, test_data, y_test, test_pred, future_data, future_pred_all, date_column='Date')
    
    # 9. 调整未来全量预测，使测试集与未来预测衔接平滑
    future_data = adjust_future_predictions(y_test, future_data, model_all_pred_col='预测值_全量')
    
    # 10. 导出预测结果，参照 USDCNY即期汇率.ipynb 中的导出方法（分别生成日度和月度数据）
    daily_output_path = "eta/USDCNY即期汇率_合并数据.xlsx"
    monthly_output_path = "eta/USDCNY即期汇率_月度数据.xlsx"
    export_forecast_results(train_data, test_data, future_data, y_train, y_test, train_pred, test_pred, future_pred_all,
                            daily_output_path, monthly_output_path, date_column='Date', target_column='真实值')
    print("结果已导出。")