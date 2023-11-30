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

    # Perform the search
    results = searcher.search(parsed_query, limit=5)
    print(results)

    # Transform the query into a TF-IDF vector using the same vectorizer
    query_vector = vectorizer.transform([user_query])
    print(query_vector)

    for hit in results:
        print(hit)
        similarity_score = cosine_similarity(query_vector, tfidf_matrix[hit.rank])[0][0]
        print(f"Similarity Score: {similarity_score}")
        original_paper_path = hit['path']

        file_name = os.path.basename(original_paper_path)
        destination_path = os.path.join(result_directory, file_name)

        shutil.copyfile(original_paper_path, destination_path)
        print(f"Paper copied to: {destination_path}")
