import json


# function to get the length or size of a data
def get_data_size(file_path):
    json_objects_count = 0

    with open(file_path, 'r') as json_file:
        for line in json_file:
            if line.strip():  # ignore empty lines
                json_objects_count += 1

    print(f"The JSON file contains {json_objects_count} JSON objects.")


# function to load the json data
def load_data(file_path, subset_size):
    papers_data = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if i >= subset_size:
                break
            papers_data.append(json.loads(line))

    return papers_data
