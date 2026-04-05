import pickle
from src.utils.reading_files import read_tsv_to_dataframe

def get_index_user(df, col, file_path):
    # Get the unique values of the 'Category' column
    unique_values = df[col].unique()
    # Create a dictionary mapping each unique value to an index
    unique_dict = {value: index for index, value in enumerate(unique_values)}
    # Specify the filename for the .pkl file
    # Save the dictionary to a .pkl file
    with open(file_path, 'wb') as file:
        pickle.dump(unique_dict, file)
    print(f"Dictionary saved to {file_path}")

def get_index_news(df, col, file_path):
    # Get the unique values of the 'Category' column
    unique_values = df[col].unique()
    # Create a dictionary mapping each unique value to an index
    unique_dict = {value: index+1 for index, value in enumerate(unique_values)}
    unique_dict ['None']=0
    # Specify the filename for the .pkl file
    # Save the dictionary to a .pkl file
    with open(file_path, 'wb') as file:
        pickle.dump(unique_dict, file)
    print(f"Dictionary saved to {file_path}")

