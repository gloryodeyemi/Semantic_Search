from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
import os
import nltk
import shutil
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# 1. Data Preprocessing
data_folder = 'data/data_new'
result_directory = 'results'
if not os.path.exists(result_directory):
    os.makedirs(result_directory)
papers = []  # Store processed text and file paths


def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())

    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    return " ".join(lemmatized_tokens)


for root, dirs, files in os.walk(data_folder):
    for file in files:
        if file.endswith(".txt"):
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                text = f.read()
                # Preprocess text (e.g., remove stopwords, stemming)
                processed_text = preprocess_text(text)
                papers.append({'path': os.path.join(root, file), 'text': processed_text})

# 2. TF-IDF Vectorization
texts = [paper['text'] for paper in papers]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

# 3. Indexing (Whoosh)
schema = Schema(path=ID(stored=True), content=TEXT)
index = create_in("academic_papers_dir", schema)
writer = index.writer()
for paper in papers:
    writer.add_document(path=paper['path'], content=paper['text'])
writer.commit()

# 4. Query Processing
query = input("Please enter a query: ")
query_vector = vectorizer.transform([preprocess_text(query)])

# 5. Retrieve Top Results
with index.searcher() as searcher:
    query_parser = QueryParser("content", index.schema)
    parsed_query = query_parser.parse(query)
    results = searcher.search(parsed_query, limit=5)

    for hit in results:
        similarity_score = cosine_similarity(query_vector, tfidf_matrix[hit.rank])[0][0]
        print(f"Similarity Score: {similarity_score}")
        original_paper_path = hit['path']

        # Extracting filename from the path
        file_name = os.path.basename(original_paper_path)

        # Creating the destination path in the result directory
        destination_path = os.path.join(result_directory, file_name)

        # Copy the file to the result directory
        shutil.copyfile(original_paper_path, destination_path)
        print(f"Paper copied to: {destination_path}")
