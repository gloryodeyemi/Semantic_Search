import os
from FAISS.utils import helper
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

DATA_PATH = '../../SentEval/data/downstream'


def read_and_process_data(data_path_neg, data_path_pos):
    with open(data_path_neg, 'r', encoding='utf-8') as f:
        neg_reviews = [line.strip() for line in f.readlines()]

    with open(data_path_pos, 'r', encoding='utf-8') as f:
        pos_reviews = [line.strip() for line in f.readlines()]

    # combine and label the data
    data = neg_reviews + pos_reviews
    labels = np.concatenate((np.zeros(len(neg_reviews)), np.ones(len(pos_reviews))))
    return data, labels


def merge_and_process_data(data_path_train, data_path_test, data_path_dev):
    # load train, test, and dev datasets
    train_data = pd.read_csv(data_path_train, sep='\t', header=None, names=['Text', 'Label'])
    test_data = pd.read_csv(data_path_test, sep='\t', header=None, names=['Text', 'Label'])
    dev_data = pd.read_csv(data_path_dev, sep='\t', header=None, names=['Text', 'Label'])

    # combine datasets
    combined_data = pd.concat([train_data, test_data, dev_data], ignore_index=True)

    # split data and labels
    data = combined_data['Text']
    labels = combined_data['Label']
    return data, labels


def load_and_process_data(data_path_train, data_path_test):
    # load train and test datasets
    try:
        train_data = pd.read_csv(data_path_train, delimiter='\t', on_bad_lines='skip')
        test_data = pd.read_csv(data_path_test, delimiter='\t', on_bad_lines='skip')
    except pd.errors.ParserError as e:
        print(f"Error reading file: {e}")
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()

    # combine datasets
    combined_data = pd.concat([train_data, test_data], ignore_index=True)

    # combine the two columns into a single column
    combined_data['Combined_Text'] = combined_data['#1 String'] + " " + combined_data['#2 String']

    # split data and labels
    data = combined_data['Combined_Text'].tolist()
    labels = combined_data['Quality']
    print(f"Data length: {len(data)}")
    print(f"Labels length: {len(labels)}")
    print(f"Sample text: {data[:5]}")
    return data, labels


def combine_and_process_data(data_path1, data_path2):
    # read the dataset
    with open(data_path1, 'r') as file:
        lines1 = file.readlines()
        print(f"Line 1 length: {len(lines1)}")

    with open(data_path2, 'r') as file:
        lines2 = file.readlines()
        print(f"Line 2 length: {len(lines2)}")

    lines = lines1 + lines2
    print(f"Extended line length: {len(lines)}")

    # extract categories and questions
    categories = [line.split()[0] for line in lines]
    questions = [' '.join(line.split()[1:]) for line in lines]

    # create a DataFrame
    data = pd.DataFrame({
        'Text': questions,
        'Category': categories
    })

    # extract labels from categories
    labels = [category.split(':')[0] for category in categories]
    data['Label'] = labels

    return data


def evaluate(data, labels, model_name, task):
    print(f"Evaluating {model_name} for {task} task...")
    model, tokenizer = helper.load_and_return(model_name)
    embeddings = np.array([helper.get_embeddings(str(review), model, tokenizer)[0] for review in data])
    embeddings_flattened = embeddings.reshape(embeddings.shape[0], -1)

    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000)

    # 10-fold cross-validation
    cv_scores_accuracy = cross_val_score(logreg, embeddings_flattened, labels, cv=10, scoring='accuracy')
    cv_scores_f1_macro = cross_val_score(logreg, embeddings_flattened, labels, cv=10, scoring='f1_macro')
    cv_scores_f1_weighted = cross_val_score(logreg, embeddings_flattened, labels, cv=10, scoring='f1_weighted')

    # calculate mean scores
    accuracy = round(np.mean(cv_scores_accuracy), 3)
    f1_macro = round(np.mean(cv_scores_f1_macro), 3)
    f1_weighted = round(np.mean(cv_scores_f1_weighted), 3)

    # update CSV file
    results = pd.DataFrame({
        'Model': [model_name.upper()],
        'Task': [task],
        'Accuracy': [accuracy],
        'Macro_F1': [f1_macro],
        'Weighted_F1': [f1_weighted]
    })

    results.to_csv('../results/embedding_evaluation_results.csv', mode='a',
                   header=not os.path.exists('../results/embedding_evaluation_results.csv'), index=False)
    print(f"Evaluation done and saved.")


# Movie Review Task
mr_data, mr_labels = read_and_process_data(f'{DATA_PATH}/MR/rt-polarity.neg', f'{DATA_PATH}/MR/rt-polarity.pos')
evaluate(mr_data, mr_labels, 'bert-base-uncased', 'MR')
evaluate(mr_data, mr_labels, 'sbert', 'MR')
print("*"*50)

# Customer Review Task
cr_data, cr_labels = read_and_process_data(f'{DATA_PATH}/CR/custrev.neg', f'{DATA_PATH}/CR/custrev.pos')
evaluate(cr_data, cr_labels, 'bert-base-uncased', 'CR')
evaluate(cr_data, cr_labels, 'sbert', 'CR')
print("*"*50)

# SUBJ Task
subj_data, subj_labels = read_and_process_data(f'{DATA_PATH}/SUBJ/subj.objective', f'{DATA_PATH}/SUBJ/subj.subjective')
evaluate(subj_data, subj_labels, 'bert-base-uncased', 'SUBJ')
evaluate(subj_data, subj_labels, 'sbert', 'SUBJ')
print("*"*50)

# MPQA Task
mpqa_data, mpqa_labels = read_and_process_data(f'{DATA_PATH}/MPQA/mpqa.neg', f'{DATA_PATH}/MPQA/mpqa.pos')
evaluate(mpqa_data, mpqa_labels, 'bert-base-uncased', 'MPQA')
evaluate(mpqa_data, mpqa_labels, 'sbert', 'MPQA')
print("*"*50)

# SST Task
sst_data, sst_labels = merge_and_process_data(f'{DATA_PATH}/SST/binary/sentiment-train',
                                              f'{DATA_PATH}/SST/binary/sentiment-test',
                                              f'{DATA_PATH}/SST/binary/sentiment-dev')
evaluate(sst_data, sst_labels, 'bert-base-uncased', 'SST')
evaluate(sst_data, sst_labels, 'sbert', 'SST')
print("*"*50)

# TREC Task
trec_data = combine_and_process_data(f'{DATA_PATH}/TREC/.!20834!train_5500.label',
                                              f'{DATA_PATH}/TREC/TREC_10.label-e')
evaluate(trec_data['Text'], trec_data['Label'], 'bert-base-uncased', 'TREC')
evaluate(trec_data['Text'], trec_data['Label'], 'sbert', 'TREC')
print("*"*50)

# MRPC Task
mrpc_data, mrpc_labels = load_and_process_data(f'{DATA_PATH}/MRPC/msr_paraphrase_train.txt',
                                              f'{DATA_PATH}/MRPC/msr_paraphrase_test.txt')
evaluate(mrpc_data, mrpc_labels, 'bert-base-uncased', 'MRPC')
evaluate(mrpc_data, mrpc_labels, 'sbert', 'MRPC')
print("*"*50)
