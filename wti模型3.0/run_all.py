#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import sys


'''
# 按照指定顺序的文件列表
files = [
    "wti模型3.0/1.1美国RBOB汽油裂解api.ipynb",
    "wti模型3.0/1.2美国RBOB汽油裂解.ipynb",
    "wti模型3.0/2.1WTI_连1-连4月差-残差项api.ipynb",
    "wti模型3.0/2.2WTI_连1-连4月差-残差项.ipynb",
    "wti模型3.0/3.1wti_连1-4api.ipynb",
    "wti模型3.0/3.2wti_连1-4.ipynb",
    "wti模型3.0/4.1wti_残差项api.ipynb",
    "wti模型3.0/4.2wti_残差项.ipynb",
    "wti模型3.0/5.1wti_原油合约价格api.ipynb",
    "wti模型3.0/5.3wti_原油合约价格_final_日度.ipynb",
    "wti模型3.0/5.4wti_原油合约价格_final_月度.ipynb",
    "wti模型3.0/6.1Brent-WTI价差api.ipynb",
    "wti模型3.0/6.2Brent-WTI价差.ipynb",


#    "wti模型3.0/7.1Brent-Dubai_api.ipynb",
#    "wti模型3.0/7.2Brent-Dubai.ipynb",
#    "wti模型3.0/8.1迪拜油_api.ipynb",
#    "wti模型3.0/8.2迪拜油.ipynb"    
]
'''


files = [
    "wti模型3.0/1.1美国RBOB汽油裂解api.ipynb",
    "wti模型3.0/1.2美国RBOB汽油裂解.ipynb",
    
    "wti模型3.0/1.3美国取暖油裂解_api.ipynb",
    "wti模型3.0/1.4美国取暖油裂解.ipynb",
    "wti模型3.0/1.5PADD3炼厂压裂裂解_api.ipynb",
    "wti模型3.0/1.6PADD3炼厂压裂裂解.ipynb",    


    "wti模型3.0/2.3WTI_连1-连4月差-残差项2_api.ipynb",
    "wti模型3.0/2.4WTI_连1-连4月差-残差项2.ipynb",
    "wti模型3.0/3.3wti_连1-4_2_api.ipynb",
    "wti模型3.0/3.4wti_连1-4_2.ipynb",



    "wti模型3.0/4.1wti_残差项api.ipynb",
    "wti模型3.0/4.2wti_残差项.ipynb",
    "wti模型3.0/5.1wti_原油合约价格api.ipynb",
    "wti模型3.0/5.3wti_原油合约价格_final_日度.ipynb",
    "wti模型3.0/5.4wti_原油合约价格_final_月度.ipynb",
    "wti模型3.0/6.1Brent-WTI价差api.ipynb",
    "wti模型3.0/6.2Brent-WTI价差.ipynb",    
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
