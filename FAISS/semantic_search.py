import time
import faiss
from preprocess import Preprocess
import torch
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model
import pandas as pd
import numpy as np

data_preprocessing = Preprocess('../data/arxiv-data.json')


def load_index(index_p):
    # Load the index
    index = faiss.read_index(index_p)
    return index


# def get_query_embedding(query, model, tokenizer):
#     # Generate embeddings for the query
#     inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.pooler_output.numpy()

def get_embedding1(text, model, tokenizer):
    # Generate embeddings for the query
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    text_embeddings = outputs.pooler_output.numpy()
    return text_embeddings


def get_embedding2(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    # text_embeddings = outputs.last_hidden_state.numpy()
    text_embeddings = outputs.last_hidden_state[:, 0].numpy()
    return text_embeddings


# function to perform semantic search
def semantic_search(query, model_name, model, tokenizer, faiss_index, faiss_dataset, top_k=5):
    # get embedding for the query
    # if model_name == "bert-base-uncased":
    #     query_embedding = get_embedding2(query, model, tokenizer)
    # elif model_name == "gpt2":
    #     query_embedding = get_embedding2(query, model, tokenizer)
    #     print(query_embedding.shape)
    #     # query_embedding = np.mean(query_embedding, axis=1, keepdims=True)
    #     # query_embedding = query_embedding.squeeze(1)
    query_embedding = get_embedding2(query, model, tokenizer)
    query_embedding = query_embedding.astype("float32")
    faiss.normalize_L2(query_embedding)

    print(f"Searching FAISS index for {model_name} model...")
    init_time = time.time()

    # search in FAISS index
    # faiss_index.nprobe = 10  # the number of clusters to search
    distance, index = faiss_index.search(query_embedding, top_k)
    print(f"Distances: {distance}\nIndices: {index}")

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


def main(user_query, model_n):
    model = None
    tokenizer = None

    # load the FAISS index and dataset
    index_path = f"faiss_index/{model_n}_faiss.index"
    dataset = pd.read_csv('../data/df-data.csv', dtype={'id': str})

    if model_n == "bert-base-uncased":
        # load the fine-tuned model and tokenizer
        model = BertModel.from_pretrained('models/bert-base-uncased_finetuned.pth')
        tokenizer = BertTokenizer.from_pretrained('models/bert-base-uncased_tokenizer')
    elif model_n == "gpt2":
        # load the fine-tuned model and tokenizer
        model = GPT2Model.from_pretrained('models/gpt2_finetuned.pth')
        tokenizer = GPT2Tokenizer.from_pretrained('models/gpt2_tokenizer')

    # perform semantic search
    result = semantic_search(user_query, model_n, model, tokenizer,
                             load_index(index_path), dataset)
    return result
