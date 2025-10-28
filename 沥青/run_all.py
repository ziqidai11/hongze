#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import sys

# 按照指定顺序的文件列表
files = [
    '沥青/1.1沥青市场价华东地区拟合残差-Brent原油期货价格_api.ipynb',
    '沥青/1.2沥青市场价华东地区拟合残差-Brent原油期货价格.ipynb',
    '沥青/1.3沥青市场价华东地区_api.ipynb',
    '沥青/1.4沥青市场价华东地区.ipynb',
    '沥青/2.1沥青月差_api.ipynb',
    '沥青/2.2沥青月差.ipynb'
    
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
