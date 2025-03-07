#!/bin/bash

# 设置矩阵大小
N=1024

# 重新编译 C++ 代码
g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

# 创建 CSV 文件
echo "t,time(ms)" > task1_results.csv

# 运行任务，t 从 1 到 20
for T in {1..20}
do
    echo "Running task1 with n=$N and t=$T..."
    RESULT=$(./task1 $N $T | tail -1)  # 获取运行时间
    echo "$T,$RESULT" >> task1_results.csv
done

# 运行 Python 代码生成 task1.pdf
python3 << EOF
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv("task1_results.csv")

# 画图
plt.figure(figsize=(8,6))
plt.plot(data["t"], data["time(ms)"], marker='o', linestyle='-')
plt.xlabel("Number of Threads (t)")
plt.ylabel("Execution Time (ms)")
plt.title("Execution Time vs. Threads for mmul")
plt.grid()
plt.savefig("task1.pdf")

print("Plot saved as task1.pdf")
EOF
