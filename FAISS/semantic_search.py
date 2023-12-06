import time
import faiss
from utils import helper, preprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# use a non-GUI backend for Matplotlib
matplotlib.use('Agg')

DATASET = pd.read_csv('data/df-data.csv', dtype={'id': str})


def load_index(index_p):
    # Load the index
    index = faiss.read_index(index_p)
    return index


# function to perform semantic search
def semantic_search(query, model_name, model, tokenizer, faiss_index, top_k=10):
    query_embedding = helper.get_embeddings(query, model, tokenizer)
    query_embedding = np.array(query_embedding).astype("float32")
    faiss.normalize_L2(query_embedding)

    print(f"Searching FAISS index for {model_name} model...")
    init_time = time.time()

    # search in FAISS index
    distance, index = faiss_index.search(query_embedding, top_k)

    # get data information for the top-k similar results
    similar_results = []
    for i, idx in enumerate(index[0]):
        if idx < len(DATASET):
            result_info = DATASET.iloc[idx]
            score = f"{distance[0][i]:.4f}"
            result_info['score'] = score
            similar_results.append(result_info.to_dict())
        else:
            print(f"Index {idx} out of bounds for the dataset.")

    # get and save the embedding time
    search_time = round(time.time() - init_time, 3)
    print(f"Searching done.\nTime taken: {search_time}s")
    preprocess.model_train_time(model_name, search_time=search_time, time_to_update="search")

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


def main(user_query, model_n):
    # load the FAISS index and dataset
    index_path = f"faiss_index/{model_n}_faiss.index"

    # load the model and tokenizer
    model, tokenizer = helper.load_and_return(model_name=model_n)

    # perform semantic search
    result = semantic_search(user_query, model_n, model, tokenizer, load_index(index_path))

    plot_search_time()
    return result
