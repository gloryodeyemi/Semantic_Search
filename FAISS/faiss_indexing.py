import time
import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from preprocess import Preprocess

data_preprocessing = Preprocess('../data/arxiv-data.json')
# load the dataset
dataset = pd.read_csv('../data/df-data.csv')


def get_embedding3(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state

    # extract the embeddings
    text_embeddings = last_hidden_states.mean(dim=1)  # mean pooling across the sequence dimension
    return text_embeddings


def load_model_and_tokenizer(model_name, tokenizer_name):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer


def build_faiss_index(embedding_list, index_file_path, data):
    embeddings = np.array(embedding_list).astype("float32")
    embeddings = np.squeeze(embeddings, axis=1)

    # set up an index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # IndexFlatIP for inner product (for cosine similarity)

    # pass the index to IndexIDMap
    index = faiss.IndexIDMap(index)
    faiss.normalize_L2(embeddings)

    # add vectors and their IDs
    index.add_with_ids(embeddings, data.index_id.values)

    # write index to a file
    faiss.write_index(index, index_file_path)

    print(f"FAISS index is built and saved to {index_file_path}")


def plot_embedding_time():
    # load the CSV file into a DataFrame
    time_data = pd.read_csv('results/training_time.csv')

    # extract necessary data
    models = time_data['Model']
    embedding_times = time_data['Embedding_time']

    # plotting the line chart
    plt.figure(figsize=(10, 6))

    for model, time_ in zip(models, embedding_times):
        plt.plot([0, 1], [0, time_], marker='o', linestyle='-', label=f"{model} = {time_:.2f}s")

    plt.title('Embedding Time for Different Models')
    plt.xlabel('Progress')
    plt.ylabel('Embedding Time')
    plt.xticks([0, 1], ['0', 'Embedding Time'])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # save the chart as a PNG image
    plt.savefig('results/embedding_times_chart.png')

    # show the plot
    plt.show()


def main(model_name):
    # load the model and tokenizer
    if model_name == "bert-base-uncased":
        model, tokenizer = load_model_and_tokenizer(model_name='models/bert-base-uncased_finetuned.pth',
                                                    tokenizer_name='models/bert-base-uncased_tokenizer')
    elif model_name == "sbert":
        model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
        model, tokenizer = load_model_and_tokenizer(model_name=model_ckpt, tokenizer_name=model_ckpt)
    elif model_name == "gpt2":
        model, tokenizer = load_model_and_tokenizer(model_name='models/gpt2_finetuned.pth',
                                                    tokenizer_name='models/gpt2_tokenizer')
        tokenizer.add_special_tokens({'pad_token': '[CLS]'})
        tokenizer.pad_token = '[CLS]'
    else:
        return

    # get the embeddings
    embeddings_list = []
    print(f"Generating {model_name} embeddings...")
    init_time = time.time()

    for _, item in dataset.iterrows():
        try:
            text_to_embed = f"{item.title} {item.abstract}"
            embedding = get_embedding3(text_to_embed, model, tokenizer)
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


# build index for bert model
main("bert-base-uncased")

# build index for sbert model
main("sbert")

# build index for gpt2 model
main("gpt2")

# plot graph
plot_embedding_time()
