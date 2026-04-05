import pandas as pd
import os
import numpy as np
import yaml
import pickle

def read_tsv_to_dataframe(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8', header=None)
        print(f"DataFrame loaded from {os.path.basename(file_path)}:")
        # print(df.head())  # Display the first few rows of the DataFrame
        return df
    else:
        print(f"{file_path} does not exist.")
        return None

def read_csv_to_dataframe(file_path):
    if os.path.exists(file_path):
        # Load the DataFrame from the CSV file
        df = pd.read_csv(file_path)
        print(f"DataFrame loaded from {os.path.basename(file_path)}:")
        # print(df.head())  # Display the first few rows of the DataFrame
        return df
    else:
        print(f"{file_path} does not exist.")
        return None

def read_npy(file_path):
    try:
        data = np.load(file_path, allow_pickle=False)
        print(f"Data loaded from {file_path}:")
        print(data)
        return data
    except FileNotFoundError:
        print(f"{file_path} does not exist.")
        return None
    except Exception as e:
        print(f"Error reading .npy file {file_path}: {e}")
        return None

def read_yaml(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            print(f"Data loaded from {file_path}:")
            print(data)
            return data
    except FileNotFoundError:
        print(f"{file_path} does not exist.")
        return None
    except yaml.YAMLError as e:
        print(f"Error reading YAML file {file_path}: {e}")
        return None

def read_pkl(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            print(f"Data loaded from {file_path}:")
            # print(data)
            return data
    except FileNotFoundError:
        print(f"{file_path} does not exist.")
        return None
    except pickle.PickleError as e:
        print(f"Error reading .pkl file {file_path}: {e}")
        return None