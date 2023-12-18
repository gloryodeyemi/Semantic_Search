# Semantic Search Engine
A semantic search engine using Facebook AI Similarity Search (FAISS) and language models (BERT and SBERT).

**Keywords:** Semantic Search, Indexing, Vectors, Embedding, Information Retrieval.

## The Dataset
A subset of the [ArXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv/) (10,000 articles) was used for this project.

## Requirements
You can find the modules and libraries used in this project in the [requirement.txt](https://github.com/gloryodeyemi/Semantic_Search/blob/main/FAISS/requirements.txt) file. You can also run the code below.
```
pip install -r requirements.txt
```

## Structure
* **[data](https://github.com/gloryodeyemi/Semantic_Search/tree/main/FAISS/data):** contains the data file used for this project.
  
* **[evaluation](https://github.com/gloryodeyemi/Semantic_Search/tree/main/FAISS/evaluation):** contains code for evaluating the models using SentEval downstream transfer and similarity tasks.

* **[utils](https://github.com/gloryodeyemi/Semantic_Search/tree/main/FAISS/utils):** contains helper functions used for the project.
  
* **[static](https://github.com/gloryodeyemi/Semantic_Search/tree/main/FAISS/static):** contains CSS and JavaScript files for the web page.

* **[templates](https://github.com/gloryodeyemi/Semantic_Search/tree/main/FAISS/templates):** contains HTML file for the web page.

* **[app.py](https://github.com/gloryodeyemi/Semantic_Search/blob/main/FAISS/app.py):** A Python file for the search engine web app using Flask.

* **[faiss_indexing.py](https://github.com/gloryodeyemi/Semantic_Search/blob/main/FAISS/faiss_indexing.py):** A Python file for setting up the FAISS index.

* **[finetune.py](https://github.com/gloryodeyemi/Semantic_Search/blob/main/FAISS/finetune.py):** A Python file for finetuning the language models.

* **[semantic_search.py](https://github.com/gloryodeyemi/Semantic_Search/blob/main/FAISS/semantic_search.py)** A Python file for the semantic search.

## Quickstart Guideline
1. Clone the repository
``` 
git clone https://github.com/gloryodeyemi/Semantic_Search.git
```
2. Change the directory to the cloned repository folder
```
%cd .../Semantic_Search/FAISS
```
3. Download the [ArXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv/) and save it to the data folder.
   
4. Install the needed packages
```
pip install -r requirements.txt
```
5. Set up the index (optional)
```
python faiss_indexing.py
```
6. Run app.py
```
python app.py
```

To run the evaluation Python files, git clone the SentEval toolkit in the project's root directory first to get them, and follow the README instructions to download the datasets.

1. Return to the project's root directory
```
cd..
```
2. Git clone SentEval toolkit
```
git clone https://github.com/facebookresearch/SentEval.git
```
3. Download datasets.

## Contact
Glory Odeyemi is undergoing her Master's program in Computer Science, Artificial Intelligence specialization at the [University of Windsor](https://www.uwindsor.ca/), Windsor, ON, Canada. You can connect with her on [LinkedIn](https://www.linkedin.com/in/glory-odeyemi-a3a680169/).

## References
1. [ArXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv/)
2. [FAISS](https://faiss.ai/index.html)
3. [BERT](https://huggingface.co/bert-base-uncased)
4. [SBERT](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1)
5. [SentEval](https://github.com/facebookresearch/SentEval)
