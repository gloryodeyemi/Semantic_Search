# load the dataset
from FAISS.preprocess import Preprocess

data_preprocessing = Preprocess('../data/arxiv-data.json')
data = data_preprocessing.convert_to_dataframe(10)
print(data.head())
