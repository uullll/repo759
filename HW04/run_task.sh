#!/bin/bash

RUNS=8

output_csv="task4_results.csv"
>"$output_csv"
echo "Run,normal,dynamic,guided,static" > "$output_csv"
for((i=1;i<=RUNS;i++))
do
    echo -n "$i," >> "$output_csv"

    ./task3  100 100 "$i" | awk -F'[: ]+' '{print $2}' | sed 's/ms//'| tr -d '\n' >> "$output_csv"
    echo -n "," >>"$output_csv"

    ./task4_dynamic 100 100 "$i" | awk -F'[: ]+' '{print $2}' | sed 's/ms//'| tr -d '\n' >>"$output_csv"
    echo -n "," >>"$output_csv"

    ./task4_guided 100 100 "$i" | awk -F'[: ]+' '{print $2}' | sed 's/ms//'| tr -d '\n'  >> "$output_csv"
    echo -n "," >>"$output_csv"

    ./task4_static 100 100 "$i" | awk -F'[: ]+' '{print $2}' | sed 's/ms//'| tr -d '\n' >>"$output_csv"
    echo "" >>"$output_csv"
    
done
echo "Execution completed"
python3 plot_task4.py
