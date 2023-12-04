import time
import faiss
from preprocess import Preprocess
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# use a non-GUI backend for Matplotlib
matplotlib.use('Agg')

data_preprocessing = Preprocess('../data/arxiv-data.json')


def load_index(index_p):
    # Load the index
    index = faiss.read_index(index_p)
    return index


def load_model_and_tokenizer(model_name, tokenizer_name):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer


def get_embedding3(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state

    # extract the embeddings
    text_embeddings = last_hidden_states.mean(dim=1)  # mean pooling across the sequence dimension
    return text_embeddings


# function to perform semantic search
def semantic_search(query, model_name, model, tokenizer, faiss_index, faiss_dataset, top_k=10):
    query_embedding = get_embedding3(query, model, tokenizer)
    query_embedding = np.array(query_embedding).astype("float32")
    # faiss.normalize_L2(query_embedding)

    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    print(f"Searching FAISS index for {model_name} model...")
    init_time = time.time()

    # search in FAISS index
    distance, index = faiss_index.search(query_embedding, top_k)
    # print(f"Distances: {distance}\nIndices: {index}")

    # get data information for the top-k similar results
    similar_results = []
    for i, idx in enumerate(index[0]):
        if idx < len(faiss_dataset):
            result_info = faiss_dataset.iloc[idx]
            score = f"{distance[0][i]:.4f}"
            result_info['score'] = score
            similar_results.append(result_info.to_dict())
        else:
            print(f"Index {idx} out of bounds for the dataset.")

    # get and save the embedding time
    search_time = round(time.time() - init_time, 3)
    print(f"Searching done.\nTime taken: {search_time}s")
    data_preprocessing.model_train_time(model_name, search_time=search_time, time_to_update="search")

    return similar_results


def plot_search_time():
    # load the CSV file into a DataFrame
    time_data = pd.read_csv('results/training_time.csv')

    # extract necessary data
    models = time_data['Model']
    search_times = time_data['Search_time']

    # plotting the line chart
    plt.figure(figsize=(10, 6))

    for model, time_ in zip(models, search_times):
        plt.plot([0, 1], [0, time_], marker='o', linestyle='-', label=f"{model} = {time_:.3f}s")

    plt.title('FAISS Search Time for Different Models')
    plt.xlabel('Progress')
    plt.ylabel('Search Time')
    plt.xticks([0, 1], ['0', 'Search Time'])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # save the chart as a PNG image
    plt.savefig('results/search_times_chart.png')

    # show the plot
    # plt.show()


def main(user_query, model_n):
    model = None
    tokenizer = None

    # load the FAISS index and dataset
    index_path = f"faiss_index/{model_n}_faiss.index"
    dataset = pd.read_csv('../data/df-data.csv', dtype={'id': str})

    # load the model and tokenizer
    if model_n == "bert-base-uncased":
        model, tokenizer = load_model_and_tokenizer(model_name='models/bert-base-uncased_finetuned.pth',
                                                    tokenizer_name='models/bert-base-uncased_tokenizer')
    elif model_n == "sbert":
        model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
        model, tokenizer = load_model_and_tokenizer(model_name=model_ckpt, tokenizer_name=model_ckpt)
    elif model_n == "gpt2":
        model, tokenizer = load_model_and_tokenizer(model_name='models/gpt2_finetuned.pth',
                                                    tokenizer_name='models/gpt2_tokenizer')
        tokenizer.add_special_tokens({'pad_token': '[CLS]'})
        tokenizer.pad_token = '[CLS]'
    else:
        return

    # perform semantic search
    result = semantic_search(user_query, model_n, model, tokenizer,
                             load_index(index_path), dataset)

    plot_search_time()
    return result
