# # Step 5: Generating PDFs for Top 5 Papers
# for i, (score, content) in enumerate(top_5_similar_papers):
#     print(f"Top Paper {i + 1} - Similarity Score: {score:.4f}")
#
#     # Display a part of the content (e.g., first 300 characters as abstract)
#     abstract = content[:300] + '...' if len(content) > 300 else content
#     print(f"Abstract: {abstract}\n")
#
#     pdf_writer = PyPDF2.PdfWriter()
#
#     # Create a new blank page
#     pdf_writer.add_blank_page(width=612, height=792)
#
#     # Convert text content to a PDF
#     buffer = io.BytesIO()
#     pdf = canvas.Canvas(buffer, pagesize=letter)
#     pdf.drawString(100, 700, content)  # Adjust coordinates and styling as needed
#     pdf.save()
#     buffer.seek(0)
#
#     # Add the generated PDF content to PdfWriter
#     pdf_reader = PyPDF2.PdfReader(buffer)
#     pdf_writer.add_page(pdf_reader.pages[0])
#
#     output_filename = f"results/top_paper_{i + 1}_score_{score:.4f}.pdf"
#     with open(output_filename, 'wb') as output:
#         pdf_writer.write(output)

# def preprocess_text(text):
#     tokens = word_tokenize(text.lower())
#     stop_words = set(stopwords.words('english'))
#     filtered_tokens = [word for word in tokens if word not in stop_words]
#     lemmatizer = WordNetLemmatizer()
#     lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
#     return " ".join(lemmatized_tokens)

import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

# Step 1: Data Loading and Preprocessing
dataset_folder = 'data/data_new'

def load_papers(dataset_folder):
    papers = []
    for root, _, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as txt_file:
                    content = txt_file.read()
                    papers.append(content)
    return papers

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

def perform_tfidf(papers_content):
    processed_papers = [preprocess_text(content) for content in papers_content]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_papers)
    return tfidf_matrix, vectorizer, processed_papers

# Load and preprocess papers
papers_content = load_papers(dataset_folder)
tfidf_matrix, vectorizer, processed_papers = perform_tfidf(papers_content)

# Step 2: Semantic Search
def semantic_search(query, processed_papers, tfidf_matrix, vectorizer):
    query = preprocess_text(query)
    query_vector = vectorizer.transform([query])

    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
    sorted_indices = similarity_scores.argsort()[0][::-1]

    top_5_papers = []
    for index in sorted_indices[:5]:
        top_5_papers.append((similarity_scores[0][index], processed_papers[index]))

    return top_5_papers

# Step 3: Get Top 5 Similar Papers
query = "Consistent hierarchies of nonlinear abstractions"
top_5_similar_papers = semantic_search(query, processed_papers, tfidf_matrix, vectorizer)
print(top_5_similar_papers)
