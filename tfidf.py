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

# Step 1: Data Loading and Preprocessing
dataset_folder = 'data/data_new'

# Index directory to store preprocessed papers
index_directory = 'index'

if not os.path.exists(index_directory):
    os.makedirs(index_directory)


def load_papers(dataset_folder):
    papers = []
    for root, dirs, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as txt_file:
                    content = txt_file.read()
                    papers.append(content)
    return papers


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


def preprocess_and_save(dataset_folder, index_directory):
    for root, dirs, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as txt_file:
                    content = txt_file.read()
                    preprocessed_content = preprocess_text(content)

                    # Save preprocessed content to index directory
                    file_name = os.path.splitext(file)[0]  # Extract file name without extension
                    index_filename = f"{file_name}_preprocessed.txt"
                    index_file_path = os.path.join(index_directory, index_filename)
                    with open(index_file_path, 'w', encoding='utf-8') as index_file:
                        index_file.write(preprocessed_content)


# Preprocess and save papers to the index directory
preprocess_and_save(dataset_folder, index_directory)


# Load and preprocess papers
papers_content = load_papers(dataset_folder)
processed_papers = [preprocess_text(content) for content in papers_content]

# Step 2: TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_papers)


# Step 3: Semantic Search
def semantic_search(query, papers_content, tfidf_matrix, vectorizer):
    query = preprocess_text(query)
    query_vector = vectorizer.transform([query])

    # Calculate cosine similarity between query vector and document vectors
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)

    # Get indices of documents sorted by similarity score
    sorted_indices = similarity_scores.argsort()[0][::-1]

    # Display top 5 similar papers
    top_5_papers = []
    for index in sorted_indices[:5]:
        top_5_papers.append((similarity_scores[0][index], papers_content[index]))

    return top_5_papers


# Step 4: Get Top 5 Similar Papers
query = "Consistent hierarchies of nonlinear abstractions"
top_5_similar_papers = semantic_search(query, papers_content, tfidf_matrix, vectorizer)

# ... (previous code remains unchanged until this point)

# Step 5: Generating txt files for Top 5 Papers
output_directory = 'results'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Step 5: Generating txt files for Top 5 Papers
output_directory = 'results'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def extract_paper_number(content):
    parts = content.split('_')
    if len(parts) >= 2:
        return parts[0]  # Extract the paper number from the content name
    return None

for i, (score, content) in enumerate(top_5_similar_papers):
    print(f"Top Paper {i + 1} - Similarity Score: {score:.4f}")

    paper_number = extract_paper_number(content)
    if paper_number:
        found = False
        for root, dirs, files in os.walk(index_directory):  # Use the index directory instead
            for file in files:
                file_paper_number = file.split('_')[0]  # Extracting paper number from filename in index directory
                if file.endswith('_preprocessed.txt') and file_paper_number == paper_number:
                    original_file_path = os.path.join(root, file)

                    # Copy the original text file to the 'results' directory
                    output_filename = f"top_paper_{i + 1}_score_{score:.4f}.txt"
                    output_file_path = os.path.join(output_directory, output_filename)
                    shutil.copy(original_file_path, output_file_path)

                    print(f"Original Text File Copied to: {output_file_path}\n")
                    found = True
                    break
            if found:
                break

        if not found:
            print(f"No text file found for paper number {paper_number}\n")
    else:
        print("Could not extract paper number from content.\n")
