#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import sys

# 按照指定顺序的文件列表
files = [


    #'汽柴煤油2.0/30.1 FU-SC_api.ipynb',
    '汽柴煤油2.0/30.2 FU-SC.ipynb',
    #'汽柴煤油2.0/31.1 LU-SC_api.ipynb',
    '汽柴煤油2.0/31.2 LU-SC.ipynb',
    #'汽柴煤油2.0/32.1 BU_SC_api.ipynb',
    '汽柴煤油2.0/32.2 BU_SC.ipynb',

    #'汽柴煤油2.0/33.1 TA_SC_api.ipynb',
    '汽柴煤油2.0/33.2 TA_SC.ipynb',

    #'汽柴煤油2.0/34.1SC期货指数_api.ipynb',
    '汽柴煤油2.0/34.2SC期货指数.ipynb',

    #'汽柴煤油2.0/35.1TA期货指数_api.ipynb',
    '汽柴煤油2.0/35.2TA期货指数.ipynb',

    #'汽柴煤油2.0/36.1 纯苯-Brent价差拟合残差-亚洲PX负荷_api.ipynb',
    '汽柴煤油2.0/36.2 纯苯-Brent价差拟合残差-亚洲PX负荷.ipynb',

    #'汽柴煤油2.0/37.1 纯苯-Brent价差_api.ipynb',
    '汽柴煤油2.0/37.2 纯苯-Brent价差.ipynb',

    #'汽柴煤油2.0/37.3 纯苯_api.ipynb',
    '汽柴煤油2.0/37.4 纯苯.ipynb',


    #'汽柴煤油2.0/38.1PX-SC_api.ipynb',
    '汽柴煤油2.0/38.2PX-SC.ipynb',

    #'汽柴煤油2.0/39.1PX_api.ipynb',
    '汽柴煤油2.0/39.2PX.ipynb',

    #'汽柴煤油2.0/40.1 EB-SC（期货指数）拟合残差-纯苯-Brent价差_api.ipynb',
    '汽柴煤油2.0/40.2 EB-SC（期货指数）拟合残差-纯苯-Brent价差.ipynb',

    #'汽柴煤油2.0/41.1 EB-SC(期货指数)_api.ipynb',
    '汽柴煤油2.0/41.2 EB-SC(期货指数).ipynb',

    #'汽柴煤油2.0/42.1EB_api.ipynb',
    '汽柴煤油2.0/42.2EB.ipynb',

    
    
    
    
    
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
