import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


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


# Example corpus
documents = [
    "This is an example document.",
    "Another document for testing purposes.",
    "One more document to add to the corpus."
]

processed_documents = [preprocess_text(doc) for doc in documents]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_documents)


def semantic_search(query, documents, tfidf_matrix, vectorizer):
    query = preprocess_text(query)
    query_vector = vectorizer.transform([query])

    # Calculate cosine similarity between query vector and document vectors
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)

    # Get indices of documents sorted by similarity score
    sorted_indices = similarity_scores.argsort()[0][::-1]

    # Display documents and their similarity scores
    for index in sorted_indices:
        print(f"Similarity Score: {similarity_scores[0][index]} - Document: {documents[index]}")


query = "example for testing"
semantic_search(query, documents, tfidf_matrix, vectorizer)
