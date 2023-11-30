# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


# def preprocess_text(text):
#     tokens = word_tokenize(text.lower())
#     stop_words = set(stopwords.words('english'))
#
#     # Lemmatization and Stemming
#     lemmatizer = WordNetLemmatizer()
#     stemmer = PorterStemmer()
#
#     processed_tokens = []
#     for word in tokens:
#         if word not in stop_words:
#             # Lemmatize and then stem the word
#             lemma = lemmatizer.lemmatize(word)
#             stemmed_word = stemmer.stem(lemma)
#             processed_tokens.append(stemmed_word)
#
#     return " ".join(processed_tokens)

# def preprocess_text(text):
#     # Tokenization
#     tokens = word_tokenize(text.lower())
#
#     # Removing stopwords
#     stop_words = set(stopwords.words('english'))
#     filtered_tokens = [word for word in tokens if word not in stop_words]
#
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
#
#     return " ".join(lemmatized_tokens)

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())

    return " ".join(tokens)

