import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from SentenceTransformer.preprocess import load_data

# Load the JSON file containing paper data
file_path = 'data/arxiv-data.json'
papers_data = load_data(file_path, 10000)

# load precomputed embeddings for the subset
paper_embeddings = np.load('SentenceTransformer/embeddings/roberta_embeddings.npy')

# use an SBERT-based model for sentence embeddings
model = SentenceTransformer('stsb-roberta-base')


# function to perform semantic search using precomputed embeddings and model
def semantic_search(query, top_k=5, batch_size=8):
    query_embedding = model.encode([query])[0]

    # calculate cosine similarity between query and precomputed paper embeddings in batches
    similarities = []
    for i in range(0, len(papers_data), batch_size):
        batch_papers = paper_embeddings[i:i + batch_size]
        batch_similarities = cosine_similarity([query_embedding], batch_papers)[0]
        similarities.extend(batch_similarities)

    # combine indices and similarities, then sort based on similarities
    results_with_indices = list(enumerate(similarities))
    results_with_indices.sort(key=lambda x: x[1], reverse=True)

    # get top k results
    top_results = results_with_indices[:top_k]

    # create a list of dictionaries containing paper info and similarity scores
    search_results = [
        {
            'paper_info': papers_data[j[0]],
            'similarity_score': j[1]
        } for j in top_results
    ]

    return search_results
