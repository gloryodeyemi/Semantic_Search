import json
import os
import pandas as pd

DATA_PATH = 'data/arxiv-data.json'
SUBSET_SIZE = 10000


# function to load the json data
def load_data():
    data = []
    with open(DATA_PATH, 'r') as file:
        for i, line in enumerate(file):
            if i >= SUBSET_SIZE:
                break
            data.append(json.loads(line))

    return data


# function to convert data into a pandas dataframe
def convert_to_dataframe():
    data = load_data()
    data_df = pd.DataFrame(data)
    # convert 'id' column to string type
    data_df['id'] = data_df['id'].astype(str)
    # adding a new column 'index_id'
    data_df['index_id'] = range(0, len(data_df))
    data_df.to_csv('data/df-data.csv', index=False)
    return data_df


def model_train_time(model_type, time_to_update, training_time=0.0, embedding_time=0.0, search_time=0.0):
    """
    Saves the model training time to a csv file.
    :param model_type: the name of the model.
    :param training_time: the model training and embedding time.
    :param embedding_time: the model embedding time.
    :param search_time: the model search time.
    :param time_to_update: the name of the time to update.
    """
    # read existing CSV file into a DataFrame
    try:
        df = pd.read_csv(f'results/training_time.csv')
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Model', 'Training_time', 'Embedding_time', 'Search_time'])

    # check if the model_name already exists in the DataFrame
    model_exists = df['Model'] == model_type

    if model_exists.any():
        if time_to_update == 'training':
            # update the training time
            df.loc[model_exists, 'Training_time'] = training_time
        elif time_to_update == 'embedding':
            # update the embedding time
            df.loc[model_exists, 'Embedding_time'] = embedding_time
        elif time_to_update == 'search':
            # update the search time
            df.loc[model_exists, 'Search_time'] = search_time
    else:
        # model doesn't exist, create a new DataFrame with the new row
        new_row = pd.DataFrame({'Model': [model_type], 'Training_time': [training_time],
                                'Embedding_time': [embedding_time], 'Search_time': [search_time]})
        # check if new_row contains any non-NA values
        if not new_row.isnull().values.all():
            df = pd.concat([df, new_row], ignore_index=True)

    # save the DataFrame back to the CSV file
    directory = "results"
    file_path = os.path.join(directory, "training_time.csv")
    if not os.path.exists(directory):
        os.makedirs(directory)
    df.to_csv(file_path, index=False)
