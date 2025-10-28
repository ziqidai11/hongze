#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import sys

# 按照指定顺序的文件列表
files = [
    "宏观经济/1.1中国10债Non-Trend_api.ipynb",
    "宏观经济/1.2中国10债Non-Trend.ipynb",
    "宏观经济/1.3中国10债Trend_api.ipynb",
    "宏观经济/1.4中国10债Trend.ipynb",
#    "宏观经济/2.1美国10债Non-trend_api.ipynb",
#    "宏观经济/2.2美国10债Non-trend.ipynb",
#    "宏观经济/2.3美国10债trend_api.ipynb",
#    "宏观经济/2.4美国10债trend.ipynb",
    "宏观经济/3.1美元指数拟合残差(10美债)_api.ipynb",
    "宏观经济/3.2美元指数拟合残差(10美债).ipynb",
    "宏观经济/3.3美元指数_api.ipynb",
    "宏观经济/3.4美元指数.ipynb",

    "宏观经济/3.5美元指数2_api.ipynb",
    "宏观经济/3.6美元指数2.ipynb",

    "宏观经济/4.1USDCNY即期汇率_api.ipynb",
    "宏观经济/4.2USDCNY即期汇率.ipynb",
#    "宏观经济/4.3人民币汇率收盘价Non-Trend_api.ipynb",
#    "宏观经济/4.3人民币汇率收盘价Non-Trend.ipynb",
#    "宏观经济/4.5人民币汇率收盘价Trend_api.ipynb",
#    "宏观经济/4.6人民币汇率收盘价Trend.ipynb",
    "宏观经济/5.1美国10年通胀预测Non-Trend_api.ipynb",
    "宏观经济/5.2美国10年通胀预测Non-Trend.ipynb",
    "宏观经济/5.3美国10年通胀预测Trend_api.ipynb",
    "宏观经济/5.4美国10年通胀预测Trend.ipynb",
    "宏观经济/6.1欧元-美元_api.ipynb",
    "宏观经济/6.2欧元-美元.ipynb",
#    "宏观经济/7.1美国GDP_api.ipynb",
#    "宏观经济/7.2美国GDP.ipynb",
    "宏观经济/8.1美国10债Non-Trend_api.ipynb",
    "宏观经济/8.2美国10债Non-Trend.ipynb",
    "宏观经济/8.3美国10债Trend_api.ipynb",
    "宏观经济/8.4美国10债Trend.ipynb",
]

def run_file(file):
    print("正在执行文件:", file)
    # 根据文件类型构造不同的命令
    if file.endswith('.py'):
        cmd = ["python", file]
    elif file.endswith('.ipynb'):
        # 使用 jupyter nbconvert 执行 notebook 文件，同步执行并直接写回原文件（避免产生新文件），使用 --inplace 参数
        cmd = ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", file]
    else:
        print("未知文件类型:", file)
        return

    try:
        # 执行命令，capture_output=True 用于捕获输出信息，check=True 如有错误会抛出 CalledProcessError
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        # 捕获到错误后，打印错误信息并退出
        print("执行文件时出现错误:", file)
        print("错误输出:", e.stderr)
        sys.exit(1)

def main():
    for file in files:
        run_file(file)
    print("所有文件执行完毕！")

if __name__ == "__main__":
    main()
