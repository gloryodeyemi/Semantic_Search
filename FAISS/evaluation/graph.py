import pandas as pd
import matplotlib.pyplot as plt


def plot_transfer(data, metric):
    # group data by Model and Task for Accuracy scores
    metric_grouped = data.groupby(['Task', 'Model'])[f'{metric}'].mean().unstack()
    ax = metric_grouped.plot(kind='bar', figsize=(12, 8))
    ax.set_ylabel(f'{metric}')
    ax.set_title(f'{metric} Scores by Model and Task')

    # add values on top of the bars
    for i, task in enumerate(metric_grouped.index):
        for j, val in enumerate(metric_grouped.iloc[i]):
            ax.text(i + (j - 0.5) * 0.2, val + 0.01, f"{round(val, 3)}", ha='center', va='bottom', fontsize=8)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'../results/{metric.lower()}.png')
    plt.show()


def plot_similarity():
    # read the data from the CSV file
    data = pd.read_csv('../results/similarity_evaluation_results.csv')

    # plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    models = data['Model']
    tasks = data['Task']
    spearman_values = data['Spearman']
    pearson_values = data['Pearson']

    bar_width = 0.35
    index = range(len(models))

    bar1 = plt.bar(index, spearman_values, bar_width, label='Spearman', color='purple')
    bar2 = plt.bar([i + bar_width for i in index], pearson_values, bar_width, label='Pearson', color='salmon')

    plt.xlabel('Model-Task Combinations')
    plt.ylabel('Correlation Values')
    plt.title('Spearman and Pearson Correlation Values by Model and Task')
    plt.xticks([i + bar_width / 2 for i in index], [f"{model}-{task}" for model, task in zip(models, tasks)])
    plt.legend()

    # displaying the values on top of bars
    for bar, values in zip((bar1, bar2), (spearman_values, pearson_values)):
        for i, value in enumerate(values):
            plt.text(bar[i].get_x() + bar[i].get_width() / 2, bar[i].get_height(),
                     f'{value}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('../results/correlation.png')
    plt.show()


# plot graph for transfer tasks
dataset = pd.read_csv('../results/embedding_evaluation_results.csv')
metrics = ['Accuracy', 'Macro_F1', 'Weighted_F1']
for metric_ in metrics:
    plot_transfer(dataset, metric_)

# plot graph for similarity tasks
plot_similarity()
