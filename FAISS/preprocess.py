import json
import pandas as pd


class Preprocess:
    def __init__(self, file_path):
        self.filepath = file_path

    # function to load the json data
    def load_data(self, subset_size):
        data = []
        with open(self.filepath, 'r') as file:
            for i, line in enumerate(file):
                if i >= subset_size:
                    break
                data.append(json.loads(line))
        # print(f"Dataset: {papers_data[:2]}")

        return data

        # function to convert papers_data into a pandas dataframe
    def convert_to_dataframe(self, subset_size):
        data = self.load_data(subset_size)
        data_df = pd.DataFrame(data)
        # convert 'id' column to string type
        data_df['id'] = data_df['id'].astype(str)
        # adding a new column 'index_id'
        data_df['index_id'] = range(0, len(data_df))
        return data_df

    def model_train_time(self, model_type, time_to_update, training_time=0.0, embedding_time=0.0, search_time=0.0):
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
        df.to_csv(f'results/training_time.csv', index=False)
