import joblib
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from text_preprocess import preprocess_text

data_folder = '../data/data_new'
papers = []  # Store processed text and file paths

for root, dirs, files in os.walk(data_folder):
    for file in files:
        if file.endswith(".txt"):
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                text = f.read()
                # Preprocess text (e.g., remove stopwords, stemming)
                processed_text = preprocess_text(text)
                print(processed_text)
                papers.append({'path': os.path.join(root, file), 'text': processed_text})

print(len(papers))
texts = [paper['text'] for paper in papers]

# Initialize TfidfVectorizer with custom preprocessing function
vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
tfidf_matrix = vectorizer.fit_transform(texts)

# Save the TF-IDF matrix
joblib.dump(tfidf_matrix, 'model/tfidf_matrix.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')

schema = Schema(path=ID(stored=True), content=TEXT)
index = create_in("academic_papers_dir", schema)
writer = index.writer()
for paper in papers:
    writer.add_document(path=paper['path'], content=paper['text'])
writer.commit()
