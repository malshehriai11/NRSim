import pandas as pd
import os
import numpy as np
import yaml
import pickle
import time

from src.utils.reading_files import *
from src.utils.newsrec_utils import prepare_hparams
from src.utils.convert2index import get_index_user, get_index_news
from src.data import newsRec_train
from src.models.nrms import NRMSModel


def combine_and_save_dfs(dfs, output_filename):
    """
    Combines multiple DataFrames into one, aligning columns if needed,
    and saves the result to a CSV file.

    Parameters:
    - dfs: list of DataFrames to combine
    - output_filename: string, name of the output CSV file
    """
    # Get all unique columns from all DataFrames
    all_columns = pd.Index([])
    for df in dfs:
        all_columns = all_columns.union(df.columns)

    # Reindex each DataFrame to ensure they all have the same columns
    aligned_dfs = [df.reindex(columns=all_columns) for df in dfs]

    # Concatenate all aligned DataFrames
    combined_df = pd.concat(aligned_dfs, ignore_index=True)
    new_column_order = ['uid', 'history', 'pos', 'neg', 'bubble']  # Replace with your actual column names
    combined_df = combined_df[new_column_order]
    combined_df = combined_df.dropna()

    # Save the combined DataFrame to CSV
    combined_df.to_csv(output_filename, index=False)
    print(f"Data saved to {output_filename}")


if __name__ == "__main__":
    yaml_file = "utils/nrms.yaml"
    wordEmb_file = "../data/hparam/embedding.npy"
    wordDict_file = "../data/hparam/word_dict.pkl"

    newsDict_file = "../data/nid2index.pkl"
    news_file = "../data/news.csv"

    userDict_file = "../data/uid2index.pkl"

    hparams = prepare_hparams(yaml_file,
                              wordEmb_file=wordEmb_file,
                              wordDict_file=wordDict_file,
                              userDict_file=userDict_file,
                              newsDict_file=newsDict_file,
                              batch_size=32,
                              epochs=5,
                              show_step=10)
    print(hparams)
    print(type(hparams))

    seed = 42
    train= 't2'

    # first training
    if train == 't1':
        behaviors_file = "../data/behaviors.csv"
        model_ws_path= "checkpoints/"+train+"/"

        # get index for users and news
        news_df = read_csv_to_dataframe(news_file)
        behaviors_df = read_csv_to_dataframe(behaviors_file)
        get_index_news(news_df, 'nid', newsDict_file)
        # get_index_user(behaviors_df, 'uid', userDict_file)



        data = newsRec_train(hparams, news_file, behaviors_file)
        history= data.input[0]
        candidate= data.input[1]
        labels= data.labels

        model = NRMSModel(hparams, seed=seed)

        # print("training:")
        model.fit([history, candidate], labels, model_ws_path, epochs=10, batch_size=512)


    # retraining
    else:

        userDict_file = "../data/uid2index.pkl"
        model_ws_path = "checkpoints/" + train + "/"

        path= '../data/rounds/t1/'
        behaviors_file = path+ "behaviors.csv"
        round_2 = path+ "round_2_behavior_df.csv"
        round_5 = path+ "round_5_behavior_df.csv"
        round_7 = path + "round_7_behavior_df.csv"
        round_10 = path + "round_10_behavior_df.csv"

        behaviors_df = read_csv_to_dataframe(behaviors_file)
        round_2_df_1 = read_csv_to_dataframe(round_2)
        round_5_df_1 = read_csv_to_dataframe(round_5)
        round_7_df_1 = read_csv_to_dataframe(round_7)
        round_10_df_1 = read_csv_to_dataframe(round_10)

        combine_and_save_dfs([behaviors_df, round_2_df_1, round_5_df_1, round_7_df_1, round_10_df_1], path+'combined_behaviors.csv')



        behaviors_file= path+'/combined_behaviors.csv'
        data = newsRec_train(hparams, news_file, behaviors_file)
        history= data.input[0]
        candidate= data.input[1]
        labels= data.labels


        model = NRMSModel(hparams, seed=seed)
        model.fit([history, candidate], labels, model_ws_path, epochs=10, batch_size=1024)











