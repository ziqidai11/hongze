#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import sys

# 按照指定顺序的文件列表
files = [
    "天然气/1.1天然气TTF连1价格Non-Trend_api.ipynb",
    "天然气/1.2天然气TTF连1价格Non-Trend.ipynb",
    "天然气/1.3天然气TTF连1价格Trend_api.ipynb", 
    "天然气/1.4天然气TTF连1价格Trend.ipynb",
    "天然气/2.1美国天然气HH连1合约Non-Trend_api.ipynb",
    "天然气/2.2美国天然气HH连1合约Non-Trend.ipynb",
    "天然气/2.3美国天然气HH连1合约_api.ipynb",
    "天然气/2.4美国天然气HH连1合约.ipynb",
    "天然气/3.1JKM-TTF_api.ipynb",
    "天然气/3.2JKM-TTF.ipynb",
    "天然气/3.3JKM_api.ipynb",
    "天然气/3.4JKM.ipynb"
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
