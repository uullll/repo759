#!/bin/bash

# 设置参数
N=1000000   # n = 10^6
TS=64       # 选择最佳 ts（你需要从之前的 task3_ts.pdf 中找到最优 ts）
OUTPUT_FILE="task3_t_results.csv"

# 重新编译 C++ 代码
g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

# 创建 CSV 文件，存储 t vs. 运行时间
echo "t,time(ms)" > $OUTPUT_FILE

# 运行任务，遍历 t=1 到 20
for T in {1..20}
do
    echo "Running task3 with n=$N, t=$T, ts=$TS..."
    RESULT=$(./task3 $N $T $TS | tail -1)  # 获取运行时间（task3 输出的最后一行）
    echo "$T,$RESULT" >> $OUTPUT_FILE
done

# 运行 Python 代码生成 task3_t.pdf
python3 << EOF
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv("$OUTPUT_FILE")

# 画图（t vs. 运行时间，线性-线性刻度）
plt.figure(figsize=(8,6))
plt.plot(data["t"], data["time(ms)"], marker='o', linestyle='-')
plt.xlabel("Number of Threads (t)")
plt.ylabel("Execution Time (ms)")
plt.title("Execution Time vs. Threads for msort (n=10^6, ts=$TS)")
plt.grid()
plt.savefig("task3_t.pdf")

print("Plot saved as task3_t.pdf")
EOF
