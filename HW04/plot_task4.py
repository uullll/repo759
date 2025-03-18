import pandas as pd
import matplotlib.pyplot as plt


# Load the results
df = pd.read_csv("task4_results.csv")

plt.figure(figsize=(10, 6))
for task in ["normal", "dynamic", "guided", "static"]:
    plt.plot(df["Run"], df[task], marker='o', label=task)

plt.xlabel("Thread Number")
plt.ylabel("Times/ms")
plt.title("Task Output Results")
plt.legend()
plt.grid(True)

plt.savefig("task4.pdf")
