import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sheet = pd.read_excel('data_input/WTI_季度.xlsx', sheet_name='Sheet1').rename(columns={'DataTime': 'Date'})
sheet.set_index('Date', inplace=True) 
sheet = sheet.reset_index().rename(columns={'index': 'Date'})

df1 = pd.ExcelFile('eta/Wti_月度_映射残差_去库幅度_合并数据.xlsx').parse('Sheet1')
df1['Date'] = pd.to_datetime(df1['Date'], errors='coerce')
df1 = df1.sort_values('Date', ascending=True)
df1 = df1.drop(columns=['实际值'])
df1 = df1.dropna() 
df1.head(1)

# 根据日期筛选数据
df2 = sheet[sheet['Date'].isin(df1['Date'])]
df2.head(1)

import pandas as pd
from typing import Tuple, Union, Optional, Literal

# ========== 工具：把"两列 DataFrame"转成 Series ==========
def _to_series(
    df_or_ser: Union[pd.DataFrame, pd.Series],
    *,
    date_col: str = "date",
    value_col: str = "value",
    name: str = "value",
) -> pd.Series:
    """
    把形如 [date, value] 的 DataFrame ⇒ Series；Series 保持原样
    """
    if isinstance(df_or_ser, pd.Series):
        return df_or_ser.rename(name)

    if not {date_col, value_col}.issubset(df_or_ser.columns):
        raise KeyError(f"DataFrame 必须包含列: {date_col}, {value_col}")

    ser = df_or_ser[[date_col, value_col]].copy()
    ser[date_col] = pd.to_datetime(ser[date_col])
    ser = ser.set_index(date_col)[value_col].astype(float).rename(name)
    return ser


# ========== 1. 计算映射 & 残差 ==========
def mapping_residual(
    y_df: Union[pd.DataFrame, pd.Series],
    x_df: Union[pd.DataFrame, pd.Series],
    y_axis: Tuple[float, float],
    x_axis: Tuple[float, float],
    lead: Union[int, pd.DateOffset] = 0,
    *,
    freq: Optional[str] = None,
    fill_method: Optional[Literal["ffill", "bfill"]] = None,
    param_method: Literal["axis", "ols"] = "axis",
    date_col: str = "date",
    value_col: str = "value",
) -> pd.DataFrame:
    """
    两列 [date, value] DataFrame 或 Series ➜ 映射值 / 残差
    其它参数说明见上一版文档
    """
    y = _to_series(y_df, date_col=date_col, value_col=value_col, name="Y")
    x = _to_series(x_df, date_col=date_col, value_col=value_col, name="X")

    if freq:
        y = y.asfreq(freq)
        x = x.asfreq(freq)
        if fill_method:
            y = y.fillna(method=fill_method)
            x = x.fillna(method=fill_method)

    # ---- 领先处理 ----
    if isinstance(lead, pd.DateOffset):
        x_shift = x.copy()
        x_shift.index = x_shift.index + lead
    else:
        x_shift = x.shift(lead)

    common_idx = y.index.intersection(x_shift.index)
    y, x_shift = y.loc[common_idx], x_shift.loc[common_idx]

    # ---- 求 a, b ----
    if param_method == "axis":
        (L1, L2), (R1, R2) = y_axis, x_axis
        a = (L2 - L1) / (R2 - R1)
        b = L2 - a * R2
    else:  # OLS
        import numpy as np

        mask = ~(y.isna() | x_shift.isna())
        X_ = np.vstack([x_shift[mask].values, np.ones(mask.sum())]).T
        a, b = np.linalg.lstsq(X_, y[mask].values, rcond=None)[0]

    # ---- 计算映射 & 残差 ----
    y_map = a * x_shift + b
    residual = y - y_map

    out = pd.DataFrame({"Y": y, "X_shift": x_shift, "Y_map": y_map, "residual": residual})
    out.attrs.update(
        {
            "slope": a,
            "intercept": b,
            "lead": lead,
            "freq": freq or y.index.inferred_freq,
            "param_method": param_method,
            "start": y.index.min(),
            "end": y.index.max(),
        }
    )
    return out


# ========== 2. 还原未来 Ŷ ==========
def restore_y(
    x_future: Union[pd.DataFrame, pd.Series, float],
    residual_pred: Union[pd.DataFrame, pd.Series, float],
    y_axis: Tuple[float, float],
    x_axis: Tuple[float, float],
    *,
    align: Literal["union", "left", "right"] = "union",
    date_col: str = "date",
    value_col: str = "value",
) -> Union[pd.Series, float]:
    """
    两列 DataFrame / Series / 标量 ➜ Ŷ
    其它参数说明见上一版文档
    """
    # ---- 若为标量，直接返回 ----
    if isinstance(x_future, (int, float)) and isinstance(residual_pred, (int, float)):
        (L1, L2), (R1, R2) = y_axis, x_axis
        a = (L2 - L1) / (R2 - R1)
        b = L2 - a * R2
        return a * float(x_future) + b + float(residual_pred)

    # ---- 转 Series，保持索引 ----
    x_series = _to_series(x_future, date_col=date_col, value_col=value_col, name="X_future")
    r_series = _to_series(residual_pred, date_col=date_col, value_col=value_col, name="residual_pred")

    # ---- 对齐索引 ----
    if align == "union":
        idx = x_series.index.union(r_series.index)
    elif align == "left":
        idx = x_series.index
    else:  # 'right'
        idx = r_series.index

    X_aligned = x_series.reindex(idx)
    R_aligned = r_series.reindex(idx)

    # ---- 线性系数 ----
    (L1, L2), (R1, R2) = y_axis, x_axis
    a = (L2 - L1) / (R2 - R1)
    b = L2 - a * R2

    return a * X_aligned + b + R_aligned

# 准备数据
x_data = pd.DataFrame({
    'Date': sheet['Date'],
    'value': sheet['EIA能源月报统计全球石油去库幅度/3MMA超季节性/3年']
})

res_data = pd.DataFrame({
    'Date': df1['Date'],
    'value': df1['WTI原油期货价格/月度/3MMA映射残差/EIA能源月报统计全球石油去库幅度/3MMA超季节性/3年（领先2月）']
})

# 设定映射轴
y_axis = (20, 120)  # WTI价格范围
x_axis = (-2, 2)    # 去库幅度范围

# 计算y hat
y_hat = restore_y(
    x_future=x_data,
    residual_pred=res_data,
    y_axis=y_axis,
    x_axis=x_axis,
    align='right',  # 使用残差数据的日期对齐
    date_col='Date',
    value_col='value'
)

# 将结果添加到数据框中
sheet['Y_hat'] = np.nan  # 首先初始化为NaN
sheet.loc[sheet['Date'].isin(y_hat.index), 'Y_hat'] = y_hat

# 打印结果，同时显示WTI原油价格进行对比
print(sheet[['Date', 'WTI原油期货价格', 'EIA能源月报统计全球石油去库幅度/3MMA超季节性/3年', 'Y_hat']].head())
