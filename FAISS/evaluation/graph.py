import pandas as pd
import matplotlib.pyplot as plt

# read the CSV file
data = pd.read_csv('../results/embedding_evaluation_results.csv')

# group data by Model and Task for Accuracy scores
accuracy_grouped = data.groupby(['Task', 'Model'])['Accuracy'].mean().unstack()
ax = accuracy_grouped.plot(kind='bar', figsize=(12, 8))
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy Scores by Model and Task')

# add values on top of the bars
for i, task in enumerate(accuracy_grouped.index):
    for j, val in enumerate(accuracy_grouped.iloc[i]):
        ax.text(i + (j - 0.5) * 0.2, val + 0.01, f"{round(val, 3)}", ha='center', va='bottom', fontsize=8)

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../results/accuracy.png')
plt.show()


# group data by Model and Task for F1 scores
f1_grouped = data.groupby(['Task', 'Model'])['Macro_F1'].mean().unstack()
ax = f1_grouped.plot(kind='bar', figsize=(12, 8))
ax.set_ylabel('F1 Score')
ax.set_title('F1 Scores by Model and Task')

# add values on top of the bars
for i, task in enumerate(f1_grouped.index):
    for j, val in enumerate(f1_grouped.iloc[i]):
        ax.text(i + (j - 0.5) * 0.2, val + 0.01, f"{round(val, 3)}", ha='center', va='bottom', fontsize=8)

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../results/f1.png')
plt.show()
