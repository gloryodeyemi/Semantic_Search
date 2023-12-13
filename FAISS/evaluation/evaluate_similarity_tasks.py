import os
import pandas as pd
from FAISS.utils import helper
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = '../../SentEval/data/downstream'


def merge_and_load_data(data_path_train, data_path_test, data_path_dev):
    # load train, test, and dev datasets
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    dev_data = pd.DataFrame()

    try:
        train_data = pd.read_csv(data_path_train, sep='\t')
    except pd.errors.ParserError as e:
        print(f"Error reading train file: {e}")

    try:
        test_data = pd.read_csv(data_path_test, sep='\t')
    except pd.errors.ParserError as e:
        print(f"Error reading test file: {e}")

    try:
        dev_data = pd.read_csv(data_path_dev, sep='\t')
    except pd.errors.ParserError as e:
        print(f"Error reading dev file: {e}")

    # combine datasets
    combined_data = pd.concat([train_data, test_data, dev_data], ignore_index=True)
    return combined_data


# function to calculate similarity using cosine similarity
def calculate_similarity(text_a, text_b, model, tokenizer):
    embedding_a = helper.get_embeddings(str(text_a), model, tokenizer)
    embedding_b = helper.get_embeddings(str(text_b), model, tokenizer)
    similarity = cosine_similarity(embedding_a, embedding_b)[0][0]
    return similarity


# function to get pearson and spearman rank correlation coefficients
def evaluate(data, model_name, task):
    print(f"Evaluating {model_name} for {task} task...")
    model, tokenizer = helper.load_and_return(model_name)
    if model_name == 'bert-base-uncased':
        model_name = 'bert'

    similarity_scores = []
    for idx, row in data.iterrows():
        similarity = calculate_similarity(row['sentence_A'], row['sentence_B'], model, tokenizer)
        similarity_scores.append(round(similarity, 3))

    print(f"Sample similarity score: {similarity_scores[:5]}")
    data[f'{model_name}_similarity'] = similarity_scores
    spearman = round(spearmanr(data[f'{model_name}_similarity'], data['relatedness_score']).correlation, 2)
    pearson = round(pearsonr(data[f'{model_name}_similarity'], data['relatedness_score'])[0], 2)

    print(f"{model_name}\n-----\nPearson: {pearson}\nSpearman: {spearman}")

    # save results to CSV
    result = pd.DataFrame({
        'Model': [model_name.upper()],
        'Task': [task],
        'Spearman': [spearman],
        'Pearson': [pearson]
    })
    result.to_csv('../results/similarity_evaluation_results.csv', mode='a',
                  header=not os.path.exists('../results/similarity_evaluation_results.csv'), index=False)
    print("Evaluation done and saved\n")


# evaluate using SICK
# dataset = merge_and_load_data(f'{DATA_PATH}/SICK/SICK_train.txt',
#                               f'{DATA_PATH}/SICK/SICK_trial.txt',
#                               f'{DATA_PATH}/SICK/SICK_test_annotated.txt')
# print(f"Dataset shape: {dataset.shape}")
# evaluate(dataset, 'bert-base-uncased', 'SICK')
# evaluate(dataset, 'sbert', 'SICK')
# print("*"*50)

# evaluate using STSBenchmark
dataset = merge_and_load_data(f'{DATA_PATH}/STS/STSBenchmark/sts-train.csv',
                              f'{DATA_PATH}/STS/STSBenchmark/sts-test.csv',
                              f'{DATA_PATH}/STS/STSBenchmark/sts-dev.csv')
print(f"Dataset shape: {dataset.shape}")
evaluate(dataset, 'bert-base-uncased', 'STSBenchmark')
evaluate(dataset, 'sbert', 'STSBenchmark')
