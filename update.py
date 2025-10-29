#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import sys
import datetime

# 按照指定顺序的文件列表
run_all_files = [
    "宏观经济/run_all.py",
    "wti模型3.0/run_all.py",
    "汽柴煤油2.0/run_all.py",
    "燃料油/run_all.py",
    "动力煤/run_all.py",
    "天然气/run_all.py",
    "黑色/玻璃/run_all.py",
    "黑色/铁矿/run_all.py", 
    "铜/run_all.py",
    "化工/run_all.py",
    "铝/run_all.py",
    "焦煤/run_all.py",
    "焦炭/run_all.py",
    "wti模型3.0/run_all2.py",
    "汽柴煤油2.0/run_all2.py",
    "汽柴煤油2.0/run_all3.py",
    "聚丙烯(PP)/run_all.py",
    "沥青/run_all.py",
    "螺纹/run_all.py",
    "汽柴煤油2.0/run_all4.py",
    
    "上传数据_日度数据_日期限制.py",
    "上传数据_列表页_数据获取.py",
    "上传数据_合并数据.py",
]

def run_file(file, current_index, total_files):
    start_time = datetime.datetime.now()
    print(f"\n{'='*50}")
    print(f"[{start_time.strftime('%H:%M:%S')}] 执行 [{current_index}/{total_files}]: {file}")
    print(f"{'='*50}\n")
    
    try:
        # 检查是否为需要实时显示输出的文件
        if "/" in file and file.endswith("run_all.py"):
            # 使用 Popen 实时显示输出
            process = subprocess.Popen(
                ["python", "-u", file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True
            )
            
            # 实时显示输出
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            # 检查错误
            if process.returncode != 0:
                error = process.stderr.read()
                print(f"\n执行失败: {file}")
                print(f"错误信息:\n{error}")
        else:
            # 普通执行方式
            result = subprocess.run(
                ["python", file],
                capture_output=True,
                text=True
            )
            print(result.stdout)
            if result.stderr:
                print(f"\n执行出现错误: {file}")
                print(f"错误信息:\n{result.stderr}")
            
        print(f"\n完成: {file}")
        print(f"耗时: {datetime.datetime.now() - start_time}")
        return True
            
    except Exception as e:
        print(f"\n执行异常: {file}")
        print(f"错误信息: {str(e)}")
        return True  # 即使出错也继续执行

def main():
    total = len(run_all_files)
    print(f"\n开始执行 {total} 个文件夹")
    
    try:
        for i, file in enumerate(run_all_files, 1):
            run_file(file, i, total)
        print("\n所有文件执行完成！")
            
    except KeyboardInterrupt:
        print("\n用户中断执行！")
        sys.exit(1)
    except Exception as e:
        print(f"\n未预期的错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
