# -*- coding: utf-8 -*-
"""
模块名称: model_pipeline
模块描述: 一个通用的XGBoost时间序列建模流水线，包含数据加载、预处理、模型训练、预测、评估、可视化以及结果导出的各个模块。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os

# 如果有其他自定义函数，这里假设你已提供；这里简单实现一个填充函数和逆序函数
def fill_missing_values(df, fill_methods, return_only_filled=False):
    """
    填充缺失值，支持不同列使用不同的方法填充。
    参数:
        df: 输入的DataFrame
        fill_methods: 字典，键为列名，值为填充方法，比如 "interpolate"
        return_only_filled: 如果为True，则只返回填充过的列，否则返回整个DataFrame
    返回:
        填充后的DataFrame
    """
    df_filled = df.copy()
    for col, method in fill_methods.items():
        if method == 'interpolate':
            df_filled[col] = df_filled[col].interpolate(method='linear')
        # 其他填充方法可以根据需要扩展
    if return_only_filled:
        return df_filled[list(fill_methods.keys())]
    else:
        return df_filled

def reverse_column(df, column):
    """
    将指定列数据逆序排列，并返回结果序列
    """
    return df[column][::-1]


# ---------------- 数据加载和预处理 ----------------

def load_data(file_path, sheet_name='Sheet1', rename_cols=None, date_column='Date', set_index=True):
    """
    从Excel文件中加载数据
    参数:
        file_path: 文件路径
        sheet_name: Excel中的sheet名称
        rename_cols: 用于重命名列的dict，例如{'DataTime': 'Date'}
        date_column: 日期列名称（加载后会转换为datetime格式）
        set_index: 是否将日期列设置为索引（默认为True）
    返回:
        加载后的DataFrame
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
    对数据进行预处理，包含以下步骤：
      1. 填充缺失值
      2. 生成提前特征（shift操作）
      3. 如需可对某些列进行逆序处理
      4. 根据日期筛选数据（基于索引或'Date'列）
      5. 按日期排序
    参数:
        df: 待处理的DataFrame（建议日期已转换为datetime，且可在索引中或包含'Date'列）
        fill_methods: dict，指定每个列的缺失值填充方法，例如 {'col1': 'interpolate'}
        shift_features: dict，指定需要创建“提前”特征，格式 { '原始列名': 提前天数 }，
                        生成的新列名称为 '原始列名_提前{n}天'
        reverse_features: list，列名列表，对这些列执行逆序操作，生成新列名称为 '列名_逆序'
        date_filter: pd.Timestamp，筛选数据的下限，若提供，则仅保留日期大于等于该值的数据
        sort_index: 是否按日期排序（默认为True）
    返回:
        预处理后的DataFrame
    """
    df_processed = df.copy()

    # 1. 填充缺失值
    if fill_methods:
        df_processed = fill_missing_values(df_processed, fill_methods, return_only_filled=False)
    
    # 2. 生成提前特征（shift操作）
    if shift_features:
        for col, shift_days in shift_features.items():
            new_col = f"{col}_提前{shift_days}天"
            df_processed[new_col] = df_processed[col].shift(shift_days)
    
    # 3. 逆序处理
    if reverse_features:
        for col in reverse_features:
            new_col = f"{col}_逆序"
            df_processed[new_col] = reverse_column(df_processed, col)
    
    # 4. 根据日期筛选数据
    if date_filter is not None:
        if df_processed.index.dtype == 'datetime64[ns]':
            df_processed = df_processed[df_processed.index >= date_filter]
        elif 'Date' in df_processed.columns:
            df_processed['Date'] = pd.to_datetime(df_processed['Date'])
            df_processed = df_processed[df_processed['Date'] >= date_filter]
    
    # 5. 索引或按日期排序
    if sort_index:
        if df_processed.index.dtype == 'datetime64[ns]':
            df_processed = df_processed.sort_index()
        elif 'Date' in df_processed.columns:
            df_processed = df_processed.sort_values('Date')
    
    return df_processed

def split_data(df, last_day, test_period=10, date_column='Date'):
    """
    将数据分割为训练集、测试集和未来数据
    参数:
        df: 输入DataFrame（日期信息可在索引中或由 date_column 指定）
        last_day: pd.Timestamp对象，表示训练集最后一条数据的日期
        test_period: 整数，测试集的样本数（默认为10）
        date_column: 如果df中未设置索引，则指定日期列名称（默认为 'Date'）
    返回:
        train_data, test_data, future_data 三个DataFrame
    """
    # 如果日期在索引中，则先把索引重置到列中
    if df.index.dtype == 'datetime64[ns]':
        df_reset = df.reset_index()
    else:
        df_reset = df.copy()
    df_reset[date_column] = pd.to_datetime(df_reset[date_column])
    
    # 分割成训练集（<= last_day）与未来数据（> last_day）
    train_data = df_reset[df_reset[date_column] <= last_day].copy()
    future_data = df_reset[df_reset[date_column] > last_day].copy()
    
    # 从训练集中分出测试集，若训练样本数大于测试期，则最后N条作为测试集
    if len(train_data) > test_period:
        test_data = train_data.iloc[-test_period:].copy()
        train_data = train_data.iloc[:-test_period].copy()
    else:
        test_data = train_data.copy()
        train_data = pd.DataFrame(columns=train_data.columns)
    
    return train_data, test_data, future_data


# ---------------- 特征标准化和样本权重计算 ----------------

def scale_features(X_train, X_test, X_future, scaler=None):
    """
    标准化特征数据
    参数:
        X_train, X_test, X_future: 输入特征（DataFrame或ndarray）
        scaler: 标准化工具（默认为StandardScaler），若为None则新建一个
    返回:
        X_train_scaled, X_test_scaled, X_future_scaled, scaler对象
    """
    if scaler is None:
        scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_future_scaled = scaler.transform(X_future)
    return X_train_scaled, X_test_scaled, X_future_scaled, scaler

def calculate_sample_weights(y, method="huber", huber_percentile=90, z_threshold=2.0):
    """
    根据目标值计算样本权重，用于降低异常值的影响
    参数:
        y: 目标变量，Series或数组
        method: "huber" 或 "zscore"，默认 "huber"
        huber_percentile: 使用Huber方法时的百分位数（默认为90）
        z_threshold: 使用zscore方法时的阈值，默认2.0
    返回:
        weights: 数组形式的样本权重
    """
    y = np.array(y)
    if method == "huber":
        residuals = np.abs(y - y.mean())
        delta = np.percentile(residuals, huber_percentile)
        weights = np.where(residuals <= delta, 1.0, delta/residuals)
    elif method == "zscore":
        std = y.std() if y.std() != 0 else 1
        z_scores = np.abs((y - y.mean()) / std)
        weights = np.where(z_scores <= z_threshold, 1.0, z_threshold/z_scores)
    else:
        weights = np.ones_like(y)
    return weights


# ---------------- XGBoost建模、训练和预测 ----------------

def train_xgb_model(X_train_scaled, y_train, X_test_scaled, y_test, params=None, 
                    num_boost_round=5000, early_stopping_rounds=100, verbose_eval=100,
                    sample_weight_method="huber", **weight_kwargs):
    """
    训练XGBoost模型
    参数:
        X_train_scaled: 训练集特征（已标准化）
        y_train: 训练集目标值
        X_test_scaled: 测试集特征（已标准化）
        y_test: 测试集目标值
        params: xgboost参数字典（若为None则使用默认参数）
        num_boost_round: 最大迭代轮数
        early_stopping_rounds: 早停轮数
        verbose_eval: 每隔多少轮打印一次信息
        sample_weight_method: 计算样本权重的方法，"huber"或"zscore"
        weight_kwargs: 计算样本权重的可选参数
    返回:
        model: 训练好的模型
        evals_result: 训练过程中的评估结果记录
        train_pred: 训练集预测值
        test_pred: 测试集预测值
    """
    if params is None:
        params = {
            'objective': 'reg:squarederror',  # 回归任务
            'learning_rate': 0.09,
            'max_depth': 7,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.9,
            'gamma': 0.05,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'eval_metric': 'rmse',
            'seed': 42,
        }
    
    # 构造DMatrix并设置样本权重
    train_weights = calculate_sample_weights(y_train, method=sample_weight_method, **weight_kwargs)
    train_dmatrix = xgb.DMatrix(X_train_scaled, label=y_train)
    train_dmatrix.set_weight(train_weights)
    
    test_dmatrix = xgb.DMatrix(X_test_scaled, label=y_test)
    
    evals_result = {}
    model = xgb.train(
        params,
        train_dmatrix,
        num_boost_round=num_boost_round,
        evals=[(test_dmatrix, 'eval'), (train_dmatrix, 'train')],
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=verbose_eval
    )
    
    train_pred = model.predict(train_dmatrix)
    test_pred = model.predict(test_dmatrix)
    return model, evals_result, train_pred, test_pred

def predict_future(model, X_future_scaled):
    """
    使用训练好的模型对未来数据进行预测
    参数:
        model: 训练好的xgboost模型
        X_future_scaled: 未来数据的标准化特征
    返回:
        future_pred: 预测结果数组
    """
    future_dmatrix = xgb.DMatrix(X_future_scaled)
    future_pred = model.predict(future_dmatrix)
    return future_pred

def evaluate_model(y_true, y_pred):
    """
    评估模型预测效果，计算均方误差 (MSE) 和判定系数 (R²)
    参数:
        y_true: 真实值
        y_pred: 预测值
    返回:
        mse, r2
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2


# ---------------- 可视化函数 ----------------

def plot_training_curve(evals_result):
    """
    绘制训练过程中的RMSE曲线
    参数:
        evals_result: xgboost训练过程中输出的评估记录字典
    """
    train_rmse = np.sqrt(evals_result['train']['rmse'])
    test_rmse = np.sqrt(evals_result['eval']['rmse'])
    
    epochs = len(train_rmse)
    x_axis = range(epochs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, train_rmse, label='训练集 RMSE', color='blue')
    plt.plot(x_axis, test_rmse, label='测试集 RMSE', color='red')
    plt.xlabel('迭代次数')
    plt.ylabel('RMSE')
    plt.title('XGBoost 训练过程中的RMSE')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(train_data, y_train, train_pred, test_data, y_test, test_pred, future_data, future_pred, date_column='Date'):
    """
    绘制训练集、测试集与未来数据的真实值和预测值对比图
    参数:
        train_data, test_data, future_data: 要求包含日期列（date_column）
        y_train, y_test: 训练集和测试集真实值
        train_pred, test_pred: 分别为训练集、测试集的预测结果
        future_pred: 未来数据预测结果
        date_column: 日期信息的列名称（如果数据集中没有设置索引，则使用指定列）
    """
    plt.figure(figsize=(30, 8))
    
    # 绘制训练集
    plt.plot(train_data[date_column], y_train, label='训练集—真实值', color='blue')
    plt.plot(train_data[date_column], train_pred, label='训练集—预测值', color='green')
    
    # 绘制测试集
    plt.plot(test_data[date_column], y_test, label='测试集—真实值', color='blue', alpha=0.7)
    plt.plot(test_data[date_column], test_pred, label='测试集—预测值', color='purple')
    
    # 绘制未来数据预测
    plt.plot(future_data[date_column], future_pred, label='未来预测', color='red')
    
    # 添加分割线标记
    if not test_data.empty:
        plt.axvline(x=test_data[date_column].iloc[0], color='black', linestyle='--', label='Train/Test Split')
    if not future_data.empty and not train_data.empty:
        last_train_date = train_data[date_column].iloc[-1]
        plt.axvline(x=last_train_date, color='black', linestyle='--', label='Future Split')
    
    plt.title('真实值与预测值对比')
    plt.legend()
    plt.grid(True)
    plt.show()


# ---------------- 结果导出函数 ----------------

def export_results(merged_df, output_path, output_format='excel', float_format='%.4f'):
    """
    将结果导出到Excel或CSV
    参数:
        merged_df: 待导出的DataFrame
        output_path: 文件输出路径
        output_format: 输出格式，可选 'excel' 或 'csv'
        float_format: 浮点数输出格式
    """
    if output_format == 'excel':
        merged_df.to_excel(output_path, index=False, float_format=float_format)
    elif output_format == 'csv':
        merged_df.to_csv(output_path, index=False, float_format=float_format)
    else:
        raise ValueError("输出格式不支持，请选择 'excel' 或 'csv'")


# ---------------- 整体Pipeline封装 ----------------

class ModelPipeline:
    """
    该类封装了整个建模流水线，从数据加载、预处理、数据分割、特征标准化、模型训练预测到结果导出。
    """
    def __init__(self, file_path, sheet_name='Sheet1', rename_cols=None, date_column='Date', set_index=True):
        """
        初始化数据加载器
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
        self.evals_result = None
        
    def load_data(self):
        """
        加载数据
        """
        self.df = load_data(self.file_path, sheet_name=self.sheet_name, rename_cols=self.rename_cols, 
                            date_column=self.date_column, set_index=self.set_index)
        return self.df
    
    def preprocess(self, fill_methods=None, shift_features=None, reverse_features=None, date_filter=None, sort_index=True):
        """
        对数据进行预处理，要求先调用load_data()
        """
        if self.df is None:
            raise ValueError("数据未加载，请先调用 load_data()")
        self.df_processed = preprocess_data(self.df, fill_methods=fill_methods, shift_features=shift_features,
                                            reverse_features=reverse_features, date_filter=date_filter, sort_index=sort_index)
        return self.df_processed
    
    def split(self, last_day, test_period=10):
        """
        按照指定的last_day分割数据为训练集、测试集和未来数据
        参数:
            last_day: pd.Timestamp对象，表示训练集最后一天
            test_period: 测试集样本数（默认为10）
        """
        self.train_data, self.test_data, self.future_data = split_data(self.df_processed, last_day, test_period, date_column=self.date_column)
        return self.train_data, self.test_data, self.future_data
    
    def prepare_and_scale(self, feature_columns, target_column):
        """
        从分割数据中提取特征和目标，并进行标准化
        参数:
            feature_columns: 列表，包含特征列的名称
            target_column: 目标变量的列名称
        返回:
            X_train_scaled, y_train, X_test_scaled, y_test, X_future_scaled
        """
        X_train = self.train_data[feature_columns]
        y_train = self.train_data[target_column]
        X_test = self.test_data[feature_columns]
        y_test = self.test_data[target_column]
        X_future = self.future_data[feature_columns]
        X_train_scaled, X_test_scaled, X_future_scaled, self.scaler = scale_features(X_train, X_test, X_future)
        return X_train_scaled, y_train, X_test_scaled, y_test, X_future_scaled
    
    def train_model(self, X_train_scaled, y_train, X_test_scaled, y_test, params=None, 
                    num_boost_round=5000, early_stopping_rounds=100, verbose_eval=100, 
                    sample_weight_method="huber", **weight_kwargs):
        """
        训练XGBoost模型，并将结果分别存储
        """
        self.model, self.evals_result, train_pred, test_pred = train_xgb_model(
            X_train_scaled, y_train, X_test_scaled, y_test, params=params,
            num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval, sample_weight_method=sample_weight_method, **weight_kwargs
        )
        return self.model, self.evals_result, train_pred, test_pred
    
    def predict_future(self, X_future_scaled):
        """
        对未来数据进行预测
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用 train_model()")
        future_pred = predict_future(self.model, X_future_scaled)
        return future_pred
    
    def export(self, merged_df, output_path, output_format='excel', float_format='%.4f'):
        """
        导出结果数据
        """
        export_results(merged_df, output_path, output_format=output_format, float_format=float_format)


# ---------------- 示例代码 ----------------
if __name__ == "__main__":
    # 示例：你需要根据实际情况修改下列参数（如文件路径等）
    file_path = "data_input/sample.xlsx"  # 请修改为你实际的文件路径
    pipeline = ModelPipeline(file_path, sheet_name='Sheet1', rename_cols={'DataTime': 'Date'}, date_column='Date')
    
    # 1. 数据加载
    df = pipeline.load_data()
    print("原始数据加载完毕。")
    
    # 2. 数据预处理（填充缺失值、提前特征创建等）
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
    reverse_features = []  # 如有需要，可指定需要逆序的列
    date_filter = pd.Timestamp('2022-11-10')  # 可按需求筛选数据起点
    
    df_processed = pipeline.preprocess(fill_methods=fill_methods, shift_features=shift_features, 
                                       reverse_features=reverse_features, date_filter=date_filter)
    print("数据预处理完毕。")
    
    # 3. 数据分割：设定 last_day（训练集最后一天），例如这里取2024-01-01
    last_day = pd.Timestamp('2024-01-01')
    train_data, test_data, future_data = pipeline.split(last_day, test_period=10)
    print("数据分割完毕。")
    
    # 4. 提取特征和目标，并进行标准化
    feature_columns = ['美国制造业PMI(预测/最新)_提前20天', '美国经济惊喜指数_提前45天', 'COMEX黄金价格Non-Trend/F0.02_提前50天']
    target_column = '美元指数拟合残差/10年期美国国债收益率'
    
    X_train_scaled, y_train, X_test_scaled, y_test, X_future_scaled = pipeline.prepare_and_scale(feature_columns, target_column)
    
    # 5. 训练XGBoost模型
    model, evals_result, train_pred, test_pred = pipeline.train_model(
        X_train_scaled, y_train, X_test_scaled, y_test,
        params=None, num_boost_round=5000, early_stopping_rounds=100, verbose_eval=100, sample_weight_method="huber"
    )
    print("模型训练完毕。")
    
    # 6. 利用训练好的模型预测未来数据
    future_pred = pipeline.predict_future(X_future_scaled)
    print("未来数据预测完毕。")
    
    # 7. 可视化训练过程和预测结果
    plot_training_curve(evals_result)
    plot_predictions(train_data, y_train, train_pred, test_data, y_test, test_pred, future_data, future_pred, date_column='Date')
    
    # 8. 合并数据示例（例如将训练集、测试集真实值与未来预测结果合并）
    merged_df = pd.concat([
        pd.concat([train_data[['Date']], pd.DataFrame({target_column: y_train})], axis=1),
        pd.concat([test_data[['Date']], pd.DataFrame({target_column: y_test})], axis=1),
        pd.concat([future_data[['Date']], pd.DataFrame({'预测值': future_pred})], axis=1)
    ], axis=0)
    merged_df = merged_df.sort_values('Date', ascending=False)
    
    # 9. 导出结果（Excel或CSV均可）
    output_path = "eta/预测结果.xlsx"
    pipeline.export(merged_df, output_path, output_format='excel')
    print("结果已导出。")