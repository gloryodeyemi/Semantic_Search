import joblib
from whoosh.index import open_dir
from whoosh.qparser import QueryParser
import os
import shutil
from sklearn.metrics.pairwise import cosine_similarity

result_directory = 'tfidf_results'
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

# Load the TF-IDF matrix
tfidf_matrix = joblib.load('model/tfidf_matrix.pkl')

# Load the vectorizer to transform the query
vectorizer = joblib.load('model/vectorizer.pkl')

index = open_dir("academic_papers_dir")

with index.searcher() as searcher:
    # Define the query parser
    query_parser = QueryParser("content", index.schema)

    # Construct the query
    user_query = input("Enter your search query: ")
    parsed_query = query_parser.parse(user_query)

    # Transform the query into a TF-IDF vector using the same vectorizer
    query_vector = vectorizer.transform([user_query])

    # Calculate cosine similarity between the query vector and documents in the TF-IDF matrix
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Get the indices of top 5 similar papers
    top_similar_indices = similarity_scores.argsort()[-5:][::-1]

    # Inside the loop where documents are retrieved based on similarity scores
    for idx in top_similar_indices:
        similarity_score = similarity_scores[idx]
        idx_str = str(idx)
        # Retrieve the document by its path field
        hit = searcher.document(path=idx_str)
        if hit:
            original_paper_path = hit['path']

            file_name = os.path.basename(original_paper_path)
            destination_path = os.path.join(result_directory, file_name)

            shutil.copyfile(original_paper_path, destination_path)
            print(f"Paper copied to: {destination_path}")
        else:
            print("No similar papers.")
