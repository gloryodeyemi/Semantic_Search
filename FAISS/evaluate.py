from __future__ import absolute_import, division, unicode_literals
import sys
import pandas as pd
import numpy as np
import torch
from FAISS.utils import helper
from sklearn.linear_model import LogisticRegression

PATH_TO_SENT_EVAL = '../SentEval'
PATH_TO_DATA = '../SentEval/data'
sys.path.insert(0, PATH_TO_SENT_EVAL)
from SentEval.senteval.engine import SE
from SentEval.senteval.tools.classifier import CustomClassifier

B_MODEL, B_TOKENIZER = helper.load_and_return(model_name="bert-base-uncased")
SB_MODEL, SB_TOKENIZER = helper.load_and_return(model_name="sbert")


# set up SentEval
def prepare(params, samples):
    return


def batcher_bert(params, batch):
    batch = [' '.join(sent) for sent in batch]
    embeddings = [helper.get_embeddings(sent, B_MODEL, B_TOKENIZER).squeeze() for sent in batch]
    return embeddings


def batcher_sbert(params, batch):
    batch = [' '.join(sent) for sent in batch]
    embeddings = [helper.get_embeddings(sent, SB_MODEL, SB_TOKENIZER).squeeze() for sent in batch]
    return embeddings


def evaluate_model(model):
    # set up the tasks for evaluation
    # tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'SST5', 'TREC', 'MRPC']
    tasks = ['MR', 'TREC', 'MRPC']
    # tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']

    # # create Logistic Regression model
    # logistic_model = LogisticRegression(max_iter=1000)

    # create the SentEval evaluator
    if model == 'bert':
        se = SE(params={'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 10}, batcher=batcher_bert)
    else:
        se = SE(params={'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 10}, batcher=batcher_sbert)

    # class_params = {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}
    # se.params['classifier'] = CustomClassifier(logistic_model, class_params, None, None)

    # create an empty DataFrame to store results
    results_df = pd.DataFrame(columns=['Task', 'Dev_Accuracy', 'Accuracy', 'N_Dev', 'N_Test'])

    # iterate through tasks and evaluate
    for task in tasks:
        print(f"Evaluating {task}...")
        results = se.eval(task)
        print(f'{task}: {results}')

        # add results to the DataFrame
        results_df = results_df._append({
            'Task': task,
            'Dev_Accuracy': results['devacc'],
            'Accuracy': results['acc'],
            'N_Dev': results['ndev'],
            'N_Test': results['ntest']
        }, ignore_index=True)

    # save the DataFrame as a CSV file
    results_df.to_csv(f'results/{model}_task_results.csv', index=False)


evaluate_model('bert')
# evaluate_model('sbert')
