import pandas as pd


# function to fetch papers starting with a given letter
def fetch_papers_by_alphabet(letter):
    dataset = pd.read_csv('data/df-data.csv', dtype={'id': str})

    # filter papers by titles starting with the given letter
    filtered_papers = dataset[dataset['title'].str.startswith(letter, na=False)]

    # return the filtered papers as a list of dictionaries
    papers_list = filtered_papers.to_dict(orient='records')
    return papers_list
