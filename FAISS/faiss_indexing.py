import time
import faiss
import numpy as np
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer, GPT2Model, GPT2Tokenizer
from preprocess import Preprocess

data_preprocessing = Preprocess('../data/arxiv-data.json')


def get_embedding2(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    # text_embeddings = outputs.last_hidden_state.numpy()
    text_embeddings = outputs.last_hidden_state[:, 0].numpy()
    return text_embeddings


def build_faiss_index(embedding_list, index_file_path, data):
    embeddings = np.array(embedding_list).astype("float32")
    embeddings = np.squeeze(embeddings, axis=1)

    # set up an index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # IndexFlatIP for inner product (for cosine similarity)

    # Step 3: Pass the index to IndexIDMap
    index = faiss.IndexIDMap(index)
    faiss.normalize_L2(embeddings)

    # Step 4: Add vectors and their IDs
    index.add_with_ids(embeddings, data.index_id.values)

    # write index to a file
    faiss.write_index(index, index_file_path)

    print(f"FAISS index is built and saved to {index_file_path}")

# def build_faiss_index(embedding_list, index_file_path, data):
#     embeddings = np.array(embedding_list).astype("float32")
#     embeddings = np.squeeze(embeddings, axis=1)
#
#     # set up an index
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)  # IndexFlatIP for inner product (for cosine similarity)
#
#     # Step 3: Pass the index to IndexIDMap
#     index = faiss.IndexIDMap(index)
#     # faiss.normalize_L2(embeddings)
#
#     # Step 4: Add vectors and their IDs
#     index.add_with_ids(embeddings, data.index_id.values)
#
#     # write index to a file
#     faiss.write_index(index, index_file_path)
#
#     print(f"FAISS index is built and saved to {index_file_path}")


def main(model_name, data):
    if model_name == "bert-base-uncased":
        # load the fine-tuned model and tokenizer
        model = BertModel.from_pretrained('models/bert-base-uncased_finetuned.pth')
        tokenizer = BertTokenizer.from_pretrained('models/bert-base-uncased_tokenizer')
    elif model_name == "gpt2":
        # load the fine-tuned model and tokenizer
        model = GPT2Model.from_pretrained('models/gpt2_finetuned.pth')
        tokenizer = GPT2Tokenizer.from_pretrained('models/gpt2_tokenizer')
    else:
        return

    # get the embeddings
    embeddings_list = []
    print(f"Generating {model_name} embeddings...")
    init_time = time.time()

    for _, item in dataset.iterrows():
        try:
            text_to_embed = f"{item.title} {item.abstract}"
            # if model_name == "bert-base-uncased":
            #     embedding = get_embedding2(text_to_embed, model, tokenizer)
            #     # print(embedding.shape)
            # elif model_name == "gpt2":
            #     embedding = get_embedding2(text_to_embed, model, tokenizer)
            #     # embedding = np.mean(embedding, axis=1, keepdims=True)
            #     # embedding = embedding.squeeze(1)
            embedding = get_embedding2(text_to_embed, model, tokenizer)
            embeddings_list.append(embedding)
        except TypeError:
            print(f"TypeError: Item causing the issue: {item}")

    # get and save the embedding time
    embedding_time = round(time.time() - init_time, 3)
    print(f"Embeddings generated.\nTime taken: {embedding_time}s")
    data_preprocessing.model_train_time(model_name, embedding_time=embedding_time, time_to_update="embedding")

    # create the index
    index_path = f"faiss_index/{model_name}_faiss.index"
    build_faiss_index(embeddings_list, index_path, dataset)


# load the dataset
dataset = pd.read_csv('../data/df-data.csv')

# build index for bert model
main("bert-base-uncased", dataset)

# build index for gpt2 model
main("gpt2", dataset)
