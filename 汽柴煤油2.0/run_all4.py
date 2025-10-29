#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import sys

# 按照指定顺序的文件列表
files = [


    #'汽柴煤油2.0/43.1石脑油-Brent拟合残差_Brent_api.ipynb',
    '汽柴煤油2.0/43.2石脑油-Brent拟合残差_Brent.ipynb', 
    #'汽柴煤油2.0/43.3石脑油-Brent拟合残差_api.ipynb',
    '汽柴煤油2.0/43.4石脑油-Brent拟合残差.ipynb',
    #'汽柴煤油2.0/43.5石脑油_api.ipynb',
    '汽柴煤油2.0/43.6石脑油.ipynb',

    #'汽柴煤油2.0/44.1PP-SC拟合残差_SC_api.ipynb',
    '汽柴煤油2.0/44.2PP-SC拟合残差_SC.ipynb', 
    #'汽柴煤油2.0/44.3PP-SC_api.ipynb',
    '汽柴煤油2.0/44.4PP-SC.ipynb',
    #'汽柴煤油2.0/44.5PP_api.ipynb',
    '汽柴煤油2.0/44.6PP.ipynb',

    #'汽柴煤油2.0/45.1PE-SC拟合残差_SC_api.ipynb',
    '汽柴煤油2.0/45.2PE-SC拟合残差_SC.ipynb', 
    #'汽柴煤油2.0/45.3PE-SC_api.ipynb',
    '汽柴煤油2.0/45.4PE-SC.ipynb',
    #'汽柴煤油2.0/45.5 PE_api.ipynb',
    '汽柴煤油2.0/45.6 PE.ipynb',

    #'汽柴煤油2.0/46.1 EG-SC拟合残差_SC_api.ipynb',
    '汽柴煤油2.0/46.2 EG-SC拟合残差_SC.ipynb', 
    #'汽柴煤油2.0/46.3 EG-SC_api.ipynb',
    '汽柴煤油2.0/46.4 EG-SC.ipynb',
    #'汽柴煤油2.0/46.5 EG_api.ipynb',
    '汽柴煤油2.0/46.6 EG.ipynb',

    #'汽柴煤油2.0/47.1 纯苯-EB_api.ipynb',
    '汽柴煤油2.0/47.2 纯苯-EB.ipynb',

    #'汽柴煤油2.0/48.1 苯乙烯-纯苯价差_api.ipynb',
    '汽柴煤油2.0/48.2 苯乙烯-纯苯价差.ipynb',

    #'汽柴煤油2.0/49.1 欧洲柴油利润拟合残差_10ppm_api.ipynb',
    '汽柴煤油2.0/49.2 欧洲柴油利润拟合残差_10ppm.ipynb',
    #'汽柴煤油2.0/49.3 欧洲柴油利润_api.ipynb',
    '汽柴煤油2.0/49.4 欧洲柴油利润.ipynb',

    #'汽柴煤油2.0/50.1山东丙烯主流价-SC指数_api.ipynb',
    '汽柴煤油2.0/50.2山东丙烯主流价-SC指数.ipynb',
    #'汽柴煤油2.0/50.3山东丙烯主流价_api.ipynb',
    '汽柴煤油2.0/50.4山东丙烯主流价.ipynb',


    #'汽柴煤油2.0/51.1PG-SC拟合残差_原油指数_api.ipynb',
    '汽柴煤油2.0/51.2PG-SC拟合残差_原油指数.ipynb',
    #'汽柴煤油2.0/51.3结算价_LPG指数_api.ipynb',
    '汽柴煤油2.0/51.4结算价_LPG指数.ipynb',

    '汽柴煤油2.0/52.1 山东丙烯-LPG_api.ipynb',
    '汽柴煤油2.0/52.2 山东丙烯-LPG.ipynb',
    '汽柴煤油2.0/53.1 PP-山东丙烯_api.ipynb',
    '汽柴煤油2.0/53.2 PP-山东丙烯.ipynb',
    '汽柴煤油2.0/54.1 PP-LPG_api.ipynb',
    '汽柴煤油2.0/54.2 PP-LPG.ipynb',

    '汽柴煤油2.0/55.1 PTA-EG_api.ipynb',
    '汽柴煤油2.0/55.2 PTA-EG.ipynb',   

    '汽柴煤油2.0/56.1 PTA-PX_api.ipynb',
    '汽柴煤油2.0/56.2 PTA-PX.ipynb',
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
