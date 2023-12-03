import string
import feature_logic
from flask import Flask, render_template, request
from semantic_search import main
import speech_recognition as sr

app = Flask(__name__)


# Function to perform voice recognition and return the recognized text
def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Say something...")
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        print(f"Recognized: {query}")
        return query
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None


@app.route('/', methods=['GET', 'POST'])
def index():
    query = None
    model = None
    results = []
    if request.method == 'POST':
        if 'text-search' in request.form:  # check for text-based search
            query = request.form['query']
            model = request.form['model']
        elif 'voice-search' in request.form:  # check for voice-based search
            query = recognize_speech()
        if query:
            # perform semantic search
            results = main(query, model)

    alphabets = list(string.ascii_uppercase)
    return render_template('index.html', query=query, results=results, show_alphabets=True,
                           alphabets=alphabets, model=model)


@app.route('/papers-by-alphabet/<letter>', methods=['GET'])
def papers_by_alphabet(letter):
    paper_results = feature_logic.fetch_papers_by_alphabet(letter)

    return render_template('index.html', paper_results=paper_results, papers_by_alphabet=True, letter=letter)


if __name__ == '__main__':
    app.run(debug=True)
