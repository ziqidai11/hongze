import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import random
import warnings

from statsmodels.tsa.stattools import acf, q_stat
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller, kpss

from statsmodels.tsa.seasonal import STL
from statsmodels.nonparametric.smoothers_lowess import lowess



class ResidualTests:
    def __init__(self, residuals):
        """
        初始化，传入残差数据。
        :param residuals: 残差序列
        """
        self.residuals = residuals

    def perform_kpss_test(self):
        """
        执行 KPSS 检验，判断残差是否平稳。
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            kpss_test = kpss(self.residuals, regression='c')
            kpss_statistic = kpss_test[0]
            kpss_p_value = kpss_test[1]
            kpss_critical_values = kpss_test[3]

            print(f'KPSS Statistic: {kpss_statistic:.4f}')
            print(f'p-value: {kpss_p_value:.4f}')

            if kpss_statistic < kpss_critical_values['5%']:
                print("The residuals are stationary (fail to reject the null hypothesis).")
            else:
                print("The residuals are not stationary (reject the null hypothesis).")

            # Check for any warning
            if len(w) > 0 and issubclass(w[0].category, Warning):
                print("Warning: The test statistic is outside the range of p-values available in the look-up table. "
                      "The actual p-value is smaller than the returned value.")

    def perform_adf_test(self):
        """
        执行 ADF 检验，判断残差是否平稳。
        """
        adf_test = adfuller(self.residuals)
        adf_statistic = adf_test[0]
        p_value = adf_test[1]
        critical_values = adf_test[4]

        print(f'ADF Statistic: {adf_statistic:.4f}')
        print(f'p-value: {p_value:.4f}')

        if p_value < 0.05:
            print("The residuals are stationary (reject the null hypothesis).")
        else:
            print("The residuals are not stationary (fail to reject the null hypothesis).")

    def perform_ljungbox_test(self, lags=10):
        """
        执行 Ljung-Box 检验，判断残差是否随机（无自相关）。
        :param lags: 检验的滞后数
        """
        ljung_box_results = acorr_ljungbox(self.residuals, lags=[lags], return_df=True)
        lb_stat = ljung_box_results['lb_stat'].values[0]
        p_value = ljung_box_results['lb_pvalue'].values[0]

        print(f"Ljung-Box Statistic: {lb_stat:.4f}")
        print(f"Ljung-Box p-value: {p_value:.4f}")

        if p_value > 0.05:
            print("The residuals are random (fail to reject the null hypothesis of no autocorrelation).")
        else:
            print("The residuals are not random (reject the null hypothesis of no autocorrelation).")

    def run_all_tests(self):
        """
        执行所有残差检验。
        """
        print("Running KPSS Test:")
        self.perform_kpss_test()
        print("\nRunning ADF Test:")
        self.perform_adf_test()
        print("\nRunning Ljung-Box Test:")
        self.perform_ljungbox_test()

'''

# 导入 
from Dtool2 import ResidualTests

# 假设 residuals 是之前生成的残差序列
residuals = data['your_value_column_residual']

# 创建 ResidualTests 类的实例
tests = ResidualTests(residuals)

# 单独执行 KPSS 检验
tests.perform_kpss_test()

# 单独执行 ADF 检验
tests.perform_adf_test()

# 单独执行 Ljung-Box 检验
tests.perform_ljungbox_test(lags=10)

# 或者执行所有测试
tests.run_all_tests()

'''

class TimeSeriesDecomposer:
    def __init__(self, data, value_col, seasonal, trend, period, seasonal_deg, trend_deg, low_pass_deg, robust=False):
        """
        初始化函数，用于设置分解所需的参数和数据。
        """
        self.data = data
        self.value_col = value_col
        self.seasonal = seasonal
        self.trend = trend
        self.period = period
        self.seasonal_deg = seasonal_deg
        self.trend_deg = trend_deg
        self.low_pass_deg = low_pass_deg
        self.robust = robust
        self.result = None

    def decompose(self):
        """
        执行 STL 分解并保存结果。
        """
        stl = STL(
            self.data[self.value_col],
            seasonal=self.seasonal,
            trend=self.trend,
            period=self.period,
            low_pass=None,
            seasonal_deg=self.seasonal_deg,
            trend_deg=self.trend_deg,
            low_pass_deg=self.low_pass_deg,
            seasonal_jump=1,
            trend_jump=1,
            low_pass_jump=1,
            robust=self.robust
        )
        self.result = stl.fit()

        # 将分解的成分存储到 data 中
        self.data[f'{self.value_col}_trend'] = self.result.trend
        self.data[f'{self.value_col}_seasonal'] = self.result.seasonal
        self.data[f'{self.value_col}_residual'] = self.result.resid

    def print_tests(self):
        """
        计算并打印各种统计测试（ADF、KPSS、Ljung-Box等）。
        """
        residual_col = f'{self.value_col}_residual'
        residuals = self.data[residual_col]

        # 创建 ResidualTests 的实例，并传入残差数据
        residual_tester = ResidualTests(residuals)

        print(f"Residual Std Dev: {np.std(residuals):.4f}")
        print("\nRunning Residual Tests:")
        
        # 执行残差的所有测试
        residual_tester.run_all_tests()

    def plot_results(self):
        """
        绘制分解的结果。
        """
        trend_col = f'{self.value_col}_trend'
        seasonal_col = f'{self.value_col}_seasonal'
        residual_col = f'{self.value_col}_residual'

        plt.figure(figsize=(14, 8))

        plt.subplot(4, 1, 1)
        plt.plot(self.data.index, self.data[self.value_col], label='Original Data', color='blue')
        plt.title(f'Original Data (Robust={self.robust}, seasonal={self.seasonal}, trend={self.trend}, period={self.period})')
        plt.grid(True)

        plt.subplot(4, 1, 2)
        plt.plot(self.data.index, self.data[trend_col], label='Trend', color='green')
        plt.title(f'Trend Component ({trend_col})')
        plt.grid(True)

        plt.subplot(4, 1, 3)
        plt.plot(self.data.index, self.data[seasonal_col], label='Seasonal', color='orange')
        plt.title(f'Seasonal Component ({seasonal_col})')
        plt.grid(True)

        plt.subplot(4, 1, 4)
        plt.plot(self.data.index, self.data[residual_col], label='Residual', color='red')
        plt.title(f'Residual Component ({residual_col})')
        plt.grid(True)

        plt.tight_layout()
        plt.show()


'''
class TimeSeriesDecomposer:
    def __init__(self, data, value_col, seasonal, trend, period, seasonal_deg=1, trend_deg=1, low_pass_deg=1, robust=True):
        self.data = data
        # 如果传入的是列表，则处理多个列
        if isinstance(value_col, list):
            self.value_cols = value_col
        else:
            self.value_cols = [value_col]  # 单列转化为列表形式，方便统一处理
        self.seasonal = seasonal
        self.trend = trend
        self.period = period
        self.seasonal_deg = seasonal_deg
        self.trend_deg = trend_deg
        self.low_pass_deg = low_pass_deg
        self.robust = robust
        self.results = {}

    def decompose(self):
        """
        对每个列进行 STL 分解
        """
        from statsmodels.tsa.seasonal import STL
        
        for col in self.value_cols:
            # 进行 STL 分解
            stl = STL(self.data[col], seasonal=self.seasonal, trend=self.trend, period=self.period, robust=self.robust)
            result = stl.fit()
            self.results[col] = result
            # 将分解结果存储到原始数据中
            self.data[f'{col}_trend'] = result.trend
            self.data[f'{col}_seasonal'] = result.seasonal
            self.data[f'{col}_residual'] = result.resid
    
    def print_tests(self):
        """
        对每个分解结果执行相关测试
        """
        for col, result in self.results.items():
            print(f'测试列: {col}')
            # 例如打印 ADF 测试结果（平稳性测试）
            from statsmodels.tsa.stattools import adfuller
            adf_test = adfuller(result.resid)
            print(f'ADF Test for {col}_residual:')
            print(f'ADF Statistic: {adf_test[0]}')
            print(f'p-value: {adf_test[1]}')
            print('-' * 40)

    def plot_results(self):
        """
        对每个分解结果绘制趋势、季节性、残差图
        """
        import matplotlib.pyplot as plt
        for col, result in self.results.items():
            plt.figure(figsize=(10, 8))

            plt.subplot(3, 1, 1)
            plt.plot(self.data[col], label=f'{col} 原始数据')
            plt.title(f'{col} 时间序列分解')

            plt.subplot(3, 1, 2)
            plt.plot(result.trend, label=f'{col} 趋势', color='g')
            plt.title(f'{col} 趋势')

            plt.subplot(3, 1, 3)
            plt.plot(result.seasonal, label=f'{col} 季节性', color='b')
            plt.title(f'{col} 季节性')

            plt.tight_layout()
            plt.show()
'''

'''

decomposer = TimeSeriesDecomposer(
   data=data,
   value_col=['column1', 'column2'],  # 传入多个列名
   seasonal=7,
   trend=13,
   period=12,
   seasonal_deg=1,
   trend_deg=1,
   low_pass_deg=1,
   robust=True
)

# 执行 STL 分解
decomposer.decompose()

# 打印测试结果
decomposer.print_tests()

# 绘制结果
decomposer.plot_results()

#    else:
#        gap = 0 
#future_data[pred_col] += gap


'''


'''
from Dtool2 import TimeSeriesDecomposer

# 假设你的 DataFrame 已经存在并且叫做 data
# 其中 'value_column' 是你要分解的时间序列列
# data = ...  # 假设这个是你之前已经定义的 DataFrame

# 实例化分解器对象
decomposer = TimeSeriesDecomposer(
   data=data,
   value_col='value_column',  # 你的时间序列列名
   seasonal=7,
   trend=13,
   period=12,
   seasonal_deg=1,
   trend_deg=1,
   low_pass_deg=1,
   robust=True
)

# 执行 STL 分解 , 结果会自动保存到 data 里面
decomposer.decompose()

# 打印测试结果
decomposer.print_tests()

# 绘制结果
decomposer.plot_results()

'''


class WeeklyDataAligner:
    def __init__(self, df_resampled, base_year=2024, file_prefix='aligned_weekly'):
        """
        初始化类，设置数据和基准年份。
        
        :param df_resampled: 需要处理的周度数据 DataFrame。
        :param base_year: 用于对齐的基准年份（默认 2024 年）。
        :param file_prefix: 保存文件的前缀（默认 'aligned_weekly'）。
        """
        self.df_resampled = df_resampled
        self.base_year = base_year
        self.file_prefix = file_prefix
        self.df_resampled_aligned = None

        # 确保索引为 DatetimeIndex
        if not isinstance(self.df_resampled.index, pd.DatetimeIndex):
            self.df_resampled.index = pd.to_datetime(self.df_resampled.index)

    def create_fridays_base_year(self):
        """
        创建基准年份的周五时间序列，自动计算每年中第一个周五。
        """
        # 获取基准年份的1月1日
        first_day_of_year = pd.Timestamp(f'{self.base_year}-01-01')

        # 找到该年中的第一个周五
        first_friday = first_day_of_year + pd.offsets.Week(weekday=4)  # 4代表周五

        # 创建从第一个周五开始到该年12月最后一个周五的日期序列
        return pd.date_range(start=first_friday, end=f'{self.base_year}-12-31', freq='W-FRI')

    def align_and_adjust(self):
        """
        对齐并调整数据，使其与基准年份的周五日期对齐。
        """
        # 创建完整的基准年份的周五时间序列
        fridays_base_year = self.create_fridays_base_year()

        # 提取年份和周数
        self.df_resampled['年份'] = self.df_resampled.index.year
        self.df_resampled['周数'] = self.df_resampled.index.isocalendar().week

        # 自动获取数据中的年份范围
        years_range = range(self.df_resampled['年份'].min(), self.df_resampled['年份'].max() + 1)
        weeks_range = range(1, 53)

        # 创建全年的周数组合
        index = pd.MultiIndex.from_product([years_range, weeks_range], names=['年份', '周数'])

        # 重建索引，使得数据对齐到完整的年份和周数组合
        self.df_resampled_aligned = self.df_resampled.set_index(['年份', '周数']).reindex(index).reset_index()

        # 保留缺失值为 NaN 并对数值进行四舍五入
        self.df_resampled_aligned['trend'] = self.df_resampled_aligned['trend'].round(3)
        self.df_resampled_aligned['seasonal'] = self.df_resampled_aligned['seasonal'].round(3)
        self.df_resampled_aligned['residual'] = self.df_resampled_aligned['residual'].round(3)

        # 创建周数到基准年份日期的映射
        week_to_date_map_base_year = {i + 1: date for i, date in enumerate(fridays_base_year)}

        # 定义日期调整函数
        def adjust_dates(row):
            week = row['周数']
            year = row['年份']
            if week in week_to_date_map_base_year:
                base_date = week_to_date_map_base_year[week]
                adjusted_date = base_date.replace(year=int(year))
                return adjusted_date
            return pd.NaT

        # 应用日期调整
        self.df_resampled_aligned['日期'] = self.df_resampled_aligned.apply(adjust_dates, axis=1)

        # 移除当前日期之后的数据
        current_date = pd.Timestamp.today()
        self.df_resampled_aligned = self.df_resampled_aligned[self.df_resampled_aligned['日期'] <= current_date]

        # 设置调整后的日期为索引
        self.df_resampled_aligned.set_index('日期', inplace=True)

        # 检查并提示缺失值
        self.check_missing_values()

        return self.df_resampled_aligned

    def check_missing_values(self):
        """
        检查对齐后的数据中是否存在缺失值，并输出相关信息。
        """
        missing_values = self.df_resampled_aligned[self.df_resampled_aligned.isna().any(axis=1)]
        if not missing_values.empty:
            print(f"警告：存在缺失值，缺失的周数为：\n{missing_values[['年份', '周数']]}")

    def save_to_csv(self):
        """
        保存对齐后的数据到不同的 CSV 文件。
        """
        self.df_resampled_aligned[['trend']].to_csv(f'{self.file_prefix}_trend.csv', date_format='%Y-%m-%d')
        self.df_resampled_aligned[['seasonal']].to_csv(f'{self.file_prefix}_seasonal.csv', date_format='%Y-%m-%d')
        self.df_resampled_aligned[['residual']].to_csv(f'{self.file_prefix}_residual.csv', date_format='%Y-%m-%d')

    def process_and_save(self):
        """
        处理数据并保存到文件。
        """
        self.align_and_adjust()
        self.save_to_csv()

'''

# 导入
from Dtool2 import WeeklyDataAligner

# 假设 df_resampled 是你的数据
aligner = WeeklyDataAligner(df_resampled, base_year=2024, file_prefix='aligned_weekly')

# 对齐并调整数据
aligned_df = aligner.align_and_adjust()

# 保存结果到CSV
aligner.process_and_save()

# 打印调整后的DataFrame
print(aligned_df)

'''




class TimeSeriesAligner:
    def __init__(self, xls, start_date='2016-09-02', end_date='2024-10-04'):
        """
        初始化函数，用于设置数据源和对齐范围。

        参数:
        - xls: Excel 文件路径或文件对象
        - start_date: 对齐的开始日期
        - end_date: 对齐的结束日期
        """
        self.xls = xls
        self.start_date = start_date
        self.end_date = end_date

    def _align_data(self, sheet_name, date_freq, date_col, value_cols):
        """
        通用对齐函数，用于对齐不同频率的数据。

        参数:
        - sheet_name: Excel 表名
        - date_freq: 日期频率（'D', 'W-FRI', 'M', 'A'）
        - date_col: 数据中日期列的名称
        - value_cols: 因子值列的名称列表
        
        返回:
        - 对齐后的 DataFrame
        """
        data = pd.read_excel(self.xls, sheet_name=sheet_name, header=0)
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq=date_freq)
        aligned_data = pd.DataFrame(index=date_range)

        # 处理每一对列（日期列 + 值列）
        for i in range(0, len(data.columns), 2):
            if i + 1 < len(data.columns):  # 确保成对的列存在
                factor_name = data.columns[i]  # 因子名称
                df = data[[data.columns[i], data.columns[i + 1]]].copy()  # 提取因子的两列
                df.columns = [date_col, factor_name]  # 重新命名列
                df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d', errors='coerce')  # 转换日期格式
                df.drop_duplicates(subset=[date_col], inplace=True)  # 去掉重复日期
                df.set_index(date_col, inplace=True)  # 将日期设置为索引
                aligned_data[factor_name] = df.reindex(aligned_data.index)[factor_name]  # 对齐到指定频率的日期

        return aligned_data

    def align_daily_data(self, sheet_name='日度数据'):
        """
        对齐每日数据。
        
        参数:
        - sheet_name: Excel 表名，默认 '日度数据'
        
        返回:
        - 对齐后的每日数据 DataFrame
        """
        return self._align_data(sheet_name=sheet_name, date_freq='D', date_col='日期', value_cols=[])

    def align_weekly_data(self, sheet_name='周度数据'):
        """
        对齐每周数据。
        
        参数:
        - sheet_name: Excel 表名，默认 '周度数据'
        
        返回:
        - 对齐后的每周数据 DataFrame
        """
        return self._align_data(sheet_name=sheet_name, date_freq='W-FRI', date_col='日期', value_cols=[])

    def align_monthly_data(self, sheet_name='月度数据'):
        """
        对齐每月数据。
        
        参数:
        - sheet_name: Excel 表名，默认 '月度数据'
        
        返回:
        - 对齐后的每月数据 DataFrame
        """
        return self._align_data(sheet_name=sheet_name, date_freq='M', date_col='日期', value_cols=[])

    def align_yearly_data(self, sheet_name='年度数据'):
        """
        对齐每年数据。
        
        参数:
        - sheet_name: Excel 表名，默认 '年度数据'
        
        返回:
        - 对齐后的每年数据 DataFrame
        """
        return self._align_data(sheet_name=sheet_name, date_freq='A', date_col='日期', value_cols=[])


'''
# 假设 xls 是包含各类数据的 Excel 文件路径或文件对象
xls = 'your_file.xlsx'  # Excel 文件路径

or

file_path = 'data_input/wti.xlsx'
xls = pd.ExcelFile(file_path)

# 创建 TimeSeriesAligner 实例
aligner = TimeSeriesAligner(xls, start_date='2016-09-02', end_date='2024-10-04')

# 对齐每日数据
aligned_daily = aligner.align_daily_data(sheet_name='日度数据')

# 对齐每周数据
aligned_weekly = aligner.align_weekly_data(sheet_name='周度数据')

# 对齐每月数据
aligned_monthly = aligner.align_monthly_data(sheet_name='月度数据')

# 对齐每年数据
aligned_yearly = aligner.align_yearly_data(sheet_name='年度数据')

'''