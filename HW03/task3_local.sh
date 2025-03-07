#!/bin/bash

# 设置参数
N=1000000   # n = 10^6
T=8         # 固定线程数 t = 8

# 重新编译 C++ 代码
g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

# 创建 CSV 文件，存储 ts vs. 运行时间
echo "ts,time(ms)" > task3_ts_results.csv

# 运行任务，遍历 ts = 2^1, 2^2, ..., 2^10
for TS in 2 4 8 16 32 64 128 256 512 1024
do
    echo "Running task3 with n=$N, t=$T, ts=$TS..."
    RESULT=$(./task3 $N $T $TS | tail -1)  # 获取运行时间（task3 输出的最后一行）
    echo "$TS,$RESULT" >> task3_ts_results.csv
done

# 运行 Python 代码生成 task3_ts.pdf
python3 << EOF
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv("task3_ts_results.csv")

# 画图（使用对数刻度）
plt.figure(figsize=(8,6))
plt.plot(data["ts"], data["time(ms)"], marker='o', linestyle='-')
plt.xscale("log")  # ts 轴为 log 变换
plt.xlabel("Threshold (ts)")
plt.ylabel("Execution Time (ms)")
plt.title("Execution Time vs. Threshold for msort (t=8)")
plt.grid()
plt.savefig("task3_ts.pdf")

print("Plot saved as task3_ts.pdf")
EOF
