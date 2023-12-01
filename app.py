from flask import Flask, render_template, request
from SentenceTransformer.search_engine import semantic_search

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    query = None
    results = []
    if request.method == 'POST':
        query = request.form['query']
        if query:
            # perform semantic search
            results = semantic_search(query)

    return render_template('index.html', query=query, results=results)


if __name__ == '__main__':
    app.run(debug=True)
