#!/bin/bash

# 设置矩阵大小 n = 1024
N=1024

# 重新编译 C++ 代码
g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp

# 创建 CSV 文件，存储 t vs. 运行时间
echo "t,time(ms)" > task2_results.csv

# 遍历 t=1 到 20
for T in {1..20}
do
    echo "Running task2 with n=$N and t=$T..."
    RESULT=$(./task2 $N $T | tail -1)  # 获取运行时间（task2 输出的最后一行）
    echo "$T,$RESULT" >> task2_results.csv
done

# 运行 Python 代码生成 task2.pdf
python3 << EOF
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv("task2_results.csv")

# 画图
plt.figure(figsize=(8,6))
plt.plot(data["t"], data["time(ms)"], marker='o', linestyle='-')
plt.xlabel("Number of Threads (t)")
plt.ylabel("Execution Time (ms)")
plt.title("Execution Time vs. Threads for convolve")
plt.grid()
plt.savefig("task2.pdf")

print("Plot saved as task2.pdf")
EOF
