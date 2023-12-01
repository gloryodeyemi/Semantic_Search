from sentence_transformers import SentenceTransformer
from preprocess import load_data
import numpy as np
import os

# load the JSON file containing paper data
file_path = '../data/arxiv-data.json'
papers_data = load_data(file_path, 10000)

# use an SBERT-based model for the sentence embeddings
model = SentenceTransformer('stsb-roberta-base')


# encode and save embeddings
def encode_and_save_embeddings(model_name):
    paper_abstracts = [paper['abstract'] for paper in papers_data]
    paper_embeddings = model.encode(paper_abstracts)

    # create a directory to save the embeddings
    os.makedirs('embeddings', exist_ok=True)

    # save the embeddings
    np.save(f'embeddings/{model_name}_embeddings.npy', paper_embeddings)


if __name__ == '__main__':
    encode_and_save_embeddings('roberta')

