#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import sys

# 按照指定顺序的文件列表
files = [
#   "汽柴煤油2.0/1.1中国汽油表需api.ipynb",
#   "汽柴煤油2.0/1.2中国汽油表需.ipynb",
#    "汽柴煤油2.0/2.1中国汽油社会库存api.ipynb",
#    "汽柴煤油2.0/2.2中国汽油社会库存.ipynb",
#    "汽柴煤油2.0/3.1中国汽油主营销售库存api.ipynb",
#    "汽柴煤油2.0/3.2中国汽油主营销售库存.ipynb",
#    "汽柴煤油2.0/4.1汽油独立炼厂表需api.ipynb",
#    "汽柴煤油2.0/4.2汽油独立炼厂表需.ipynb",
#    "汽柴煤油2.0/5.1汽油独立炼厂库存api.ipynb",
#    "汽柴煤油2.0/5.2汽油独立炼厂库存.ipynb",
    "汽柴煤油2.0/5.3中国汽油需求(多因子)_api.ipynb",
    "汽柴煤油2.0/5.4中国汽油需求(多因子).ipynb",
#    "汽柴煤油2.0/6.1.1山东汽油裂解_残差api.ipynb",
#    "汽柴煤油2.0/6.1.2山东汽油裂解_残差.ipynb",
#    "汽柴煤油2.0/6.2.1山东汽油裂解api.ipynb",
#    "汽柴煤油2.0/6.2.2山东汽油裂解.ipynb",
#    "汽柴煤油2.0/7.1.1山东柴油裂解差Non-trend_api.ipynb",
#    "汽柴煤油2.0/7.1.2山东柴油裂解差Non-trend.ipynb",
#    "汽柴煤油2.0/7.1.3山东柴油裂解差Trend_api.ipynb",
#    "汽柴煤油2.0/7.1.4山东柴油裂解差Trend.ipynb",
#    "汽柴煤油2.0/7.3中国柴油需求Non-trend_api.ipynb",
#    "汽柴煤油2.0/7.3中国柴油需求Non-trend.ipynb",
#    "汽柴煤油2.0/7.3中国柴油需求Trend_api.ipynb",
#    "汽柴煤油2.0/7.3中国柴油需求Trend.ipynb",   
    "汽柴煤油2.0/7.4中国柴油需求(多因子)_api.ipynb",
    "汽柴煤油2.0/7.4中国柴油需求(多因子).ipynb",
#    "汽柴煤油2.0/8.1柴油主营销售库存_api.ipynb",
#    "汽柴煤油2.0/8.2柴油主营销售库存.ipynb",
#    "汽柴煤油2.0/9.1柴油社会库存_api.ipynb",
#    "汽柴煤油2.0/9.2柴油社会库存.ipynb",
#    "汽柴煤油2.0/10.1柴油独立炼厂表需_api.ipynb",
#    "汽柴煤油2.0/10.2柴油独立炼厂表需.ipynb",
#    "汽柴煤油2.0/11.1柴油独立炼厂产量_api.ipynb",   
#    "汽柴煤油2.0/11.2柴油独立炼厂产量.ipynb",
#    "汽柴煤油2.0/12.1柴油独立炼厂库存_api.ipynb",
#    "汽柴煤油2.0/12.2柴油独立炼厂库存.ipynb",
#    "汽柴煤油2.0/13.1柴油裂解差拟合残差-库存_api.ipynb",
#    "汽柴煤油2.0/13.2柴油裂解差拟合残差-库存.ipynb",
#    "汽柴煤油2.0/14.1柴油裂解差(拟合残差)_api.ipynb",
#    "汽柴煤油2.0/14.2柴油裂解差(拟合残差).ipynb",
    "汽柴煤油2.0/14.3山东柴油裂解差(多因子)_api.ipynb",
    "汽柴煤油2.0/14.4山东柴油裂解差(多因子).ipynb",
    "汽柴煤油2.0/14.5山东汽油裂解差(多因子)_api.ipynb",
    "汽柴煤油2.0/14.6山东汽油裂解差(多因子).ipynb",
#    "汽柴煤油2.0/15.1煤柴价差日度api.ipynb",
#    "汽柴煤油2.0/15.2煤柴价差日度.ipynb",
    "汽柴煤油2.0/15.3新加坡航空煤油裂解价差拟合残差_布伦特迪拜api.ipynb",
    "汽柴煤油2.0/15.4新加坡航空煤油裂解价差拟合残差_布伦特迪拜.ipynb",
#    "汽柴煤油2.0/15.5新加坡航空煤油裂解价差api.ipynb",
#    "汽柴煤油2.0/15.6新加坡航空煤油裂解价差.ipynb",
    "汽柴煤油2.0/15.7新加坡航空煤油裂解价差_api.ipynb",
    "汽柴煤油2.0/15.8新加坡航空煤油裂解价差.ipynb",
#    "汽柴煤油2.0/16.1Brent-汽油-柴油-煤油api.ipynb",
    "汽柴煤油2.0/16.1Brent-汽油-柴油-煤油api2.ipynb",
#    "汽柴煤油2.0/16.2Brent-汽油-柴油-煤油_预测.ipynb",
#    "汽柴煤油2.0/16.2Brent-汽油-柴油-煤油_预测2.ipynb",
    "汽柴煤油2.0/16.2Brent-汽油-柴油-煤油_预测3.ipynb",
#    "汽柴煤油2.0/27.1RBOB汽油裂解价差_api.ipynb",
#    "汽柴煤油2.0/27.2RBOB汽油裂解价差.ipynb"
  
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
