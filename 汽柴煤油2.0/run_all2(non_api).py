#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import sys

# 按照指定顺序的文件列表
files = [

    "汽柴煤油2.0/中石化航空煤油Non-Trend_F0.2.ipynb",

    "汽柴煤油2.0/中石化航空煤油Trend.ipynb",

    '汽柴煤油2.0/17.2新加坡92汽油裂解.ipynb',  

    '汽柴煤油2.0/18.2新加坡10ppm.ipynb',  

    '汽柴煤油2.0/19.2汽油出口利润(华东-新加坡).ipynb',

    '汽柴煤油2.0/20.2柴油出口利润(华东-新加坡).ipynb',

    '汽柴煤油2.0/21.2中国汽油出口计划量.ipynb',

    '汽柴煤油2.0/22.2FU连1_连2.ipynb',

    '汽柴煤油2.0/23.2原油加工量.ipynb',

    '汽柴煤油2.0/24.2LU-FU.ipynb',

    '汽柴煤油2.0/25.2FU-BU.ipynb',

    '汽柴煤油2.0/26.2LU-BU.ipynb',

    '汽柴煤油2.0/27.2RBOB汽油裂解价差.ipynb',

    '汽柴煤油2.0/28.2 SC期货指数-Brent原油期货价格.ipynb',

    '汽柴煤油2.0/29.2 SC原油连1-连3月差.ipynb'
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
