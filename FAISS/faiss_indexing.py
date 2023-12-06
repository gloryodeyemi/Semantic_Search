import os
import time
import faiss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import helper, preprocess


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


def main(model_name, data):
    # load the model and tokenizer
    model, tokenizer = helper.load_and_return(model_name=model_name)

    # get the embeddings
    embeddings_list = []
    print(f"Generating {model_name} embeddings...")
    init_time = time.time()

    for _, item in data.iterrows():
        try:
            text_to_embed = f"{item.title} {item.abstract}"
            embedding = helper.get_embeddings(text_to_embed, model, tokenizer)
            embeddings_list.append(embedding)
        except TypeError:
            print(f"TypeError: Item causing the issue: {item}")

    # get and save the embedding time
    embedding_time = round(time.time() - init_time, 3)
    print(f"Embeddings generated.\nTime taken: {embedding_time}s")
    preprocess.model_train_time(model_name, embedding_time=embedding_time, time_to_update="embedding")

    # create the index
    directory = "faiss_index"
    if not os.path.exists(directory):
        os.makedirs(directory)

    index_path = os.path.join(directory, f"{model_name}_faiss.index")
    build_faiss_index(embeddings_list, index_path, data)


# load the dataset
dataset = preprocess.convert_to_dataframe()

# build index for bert model
main("bert-base-uncased", dataset)

# build index for sbert model
main("sbert", dataset)

# plot graph
plot_embedding_time()
