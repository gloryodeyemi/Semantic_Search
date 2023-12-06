import string
from utils import feature_logic
from flask import Flask, render_template, request
from semantic_search import main

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    query = None
    model = None
    results = []
    if request.method == 'POST':
        query = request.form['query']
        model = request.form['model']
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
