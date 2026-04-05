import pickle
import numpy as np
import pandas as pd
import random
import re
from joblib import Parallel, delayed
from src.utils.reading_files import read_csv_to_dataframe, read_pkl
from src.utils.newsrec_utils import prepare_hparams

from src.evaluation.bubble_eval import (categories_distribution_info,
                                        sentiment_distribution_info,
                                        politics_distribution_info)


def get_user_cand_attribute (news_file, categories, *arrays, dynamic_user_update):

    interaction_df, user_df, candidate_df = get_user_cand_info(news_file, categories, *arrays)

    # Convert necessary columns to dictionaries for faster lookup
    user_category_dict = user_df.set_index('uid')['history_category'].astype(str).to_dict()
    user_sentiment_dict = user_df.set_index('uid')['history_sentiment'].astype(str).to_dict()
    user_politics_dict = user_df.set_index('uid')['history_politics'].astype(str).to_dict()

    candidate_category_dict = candidate_df.set_index('candidate')['candidate_category'].astype(str).to_dict()
    candidate_sentiment_dict = candidate_df.set_index('candidate')['candidate_sentiment'].astype(str).to_dict()
    candidate_politics_dict = candidate_df.set_index('candidate')['candidate_politics'].astype(str).to_dict()

    # Initialize the attribute dictionary
    # attribute_dict = {}
    attribute_list = []
    for _, row in interaction_df.iterrows():
        user = row['uid']
        candidate = str(row['candidate'])

        # Retrieve user and candidate attributes from dictionaries
        user_category_str = user_category_dict.get(user, '')
        user_sentiment_str = user_sentiment_dict.get(user, '')
        user_politics_str = user_politics_dict.get(user, '')

        cand_category_str = candidate_category_dict.get(candidate, '')
        cand_sentiment_str = candidate_sentiment_dict.get(candidate, '')
        cand_politics_str = candidate_politics_dict.get(candidate, '')

        user_ctg = categories_distribution_info(user_category_str, categories, dynamic_user_update, user)
        user_snt = sentiment_distribution_info(user_sentiment_str, dynamic_user_update, user)
        user_pol = politics_distribution_info(user_politics_str, dynamic_user_update, user)

        cand_ctg = categories_distribution_info(cand_category_str, categories)
        cand_snt = sentiment_distribution_info(cand_sentiment_str)
        cand_pol = politics_distribution_info(cand_politics_str)

        # Calculate the attributes
        temp_dict = {
            'user': user,
            'user_prob_dist': user_ctg['categories_prob_dist'],
            'user_entropy': user_ctg['categories_normalized_entropy'],
            'user_prob_snt': user_snt['sentiment_prob_dist'],
            'user_avg_snt': user_snt['average_sentiment'],
            'user_prob_pol': user_pol['politics_prob_dist'],
            'user_avg_pol': user_pol['average_politics'],

            'candidate': candidate,
            'cand_prob_dist': cand_ctg['categories_prob_dist'],
            'cand_prob_snt': cand_snt['sentiment_prob_dist'],
            'cand_prob_pol': cand_pol['politics_prob_dist'],
        }

        # attribute_dict[(user, candidate)] = temp_dict
        attribute_list.append(temp_dict)

    return attribute_list

def get_user_cand_info (news_file, categories, *arrays):
    news_df = read_csv_to_dataframe(news_file)

    bubble= arrays[0]
    uid= arrays[1]
    history= arrays[2]
    candidate= arrays[3]

    #Users:
    # Flatten the last dimension of history_array and convert to a list of strings
    history_strings = [' '.join(map(str, row.flatten())) for row in history]
    candidate_strings = [' '.join(map(str, row.flatten())) for row in candidate]

    # Create the DataFrame
    interaction_df= pd.DataFrame({
        'uid': uid.flatten(),
        'candidate': candidate.flatten()
    })

    user_df = pd.DataFrame({
        'bubble': bubble.flatten(),
        'uid': uid.flatten(),
        'history': history_strings,
    })

    candidate_df = pd.DataFrame({
        'candidate': candidate_strings
    })
    # Remove duplicated rows
    user_df = user_df.drop_duplicates(subset='uid')
    candidate_df = candidate_df.drop_duplicates(subset='candidate')

    user_df= get_user_info(user_df, news_df)
    candidate_df= get_cand_info(candidate_df, news_df)

    return interaction_df, user_df, candidate_df

# Function to replace words based on the mapping
def replace_words(text, mapping):
    for word, replacement in mapping.items():
        text = text.replace(word, replacement)
    return text

def get_user_info(user_df, news_df):
    extract_info(user_df, news_df, "history", "category")
    extract_info(user_df, news_df, "history", "sentiment")
    extract_info(user_df, news_df, "history", "politics")

    # Convert all values to lowercase:
    # user_df = user_df.applymap(lambda s: s.lower() if type(s) == str else s)
    # Apply to string columns only, convert all values to lowercase
    user_df[user_df.select_dtypes(include=['object']).columns] = user_df.select_dtypes(include=['object']).apply(
        lambda col: col.str.lower())



    # Apply the mapping to the relevant columns
    user_df['history_sentiment'] = user_df['history_sentiment'].apply(lambda x: replace_words(x, {"mixed": "neutral"}))
    user_df['history_politics'] = user_df['history_politics'].apply(lambda x: replace_words(x, {"mixed": "center", 'center-left':'center'}))

    return user_df

def get_cand_info(candidate_df, news_df):
    extract_info(candidate_df, news_df, "candidate", "category")
    extract_info(candidate_df, news_df, "candidate", "sentiment")
    extract_info(candidate_df, news_df, "candidate", "politics")

    candidate_df = candidate_df.applymap(lambda s: s.lower() if type(s) == str else s)
    # Apply the mapping to the relevant columns
    candidate_df['candidate_sentiment'] = candidate_df['candidate_sentiment'].apply(lambda x: replace_words(x, {"mixed": "neutral"}))
    candidate_df['candidate_politics'] = candidate_df['candidate_politics'].apply(
        lambda x: replace_words(x, {"mixed": "center", 'center-left': 'center'}))
    return candidate_df

def extract_info(df1, df2, c1, c2):
    # Create a dictionary mapping words to topics
    word_to_topic = dict(zip(df2['nid'], df2[c2]))

    # Function to map words to topics
    def map_words_to_topics(string, word_to_topic):
        words = str(string).split()
        topics = [word_to_topic[int(word)] for word in words if int(word) in word_to_topic]
        return ' '.join(topics)

    # Apply the function to create the hist_topic column
    df1[c1+'_'+c2] = df1[c1].apply(lambda x: map_words_to_topics(x, word_to_topic))


