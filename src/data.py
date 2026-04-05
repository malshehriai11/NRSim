import numpy as np
import pandas as pd
import random
import re
import time

from joblib import Parallel, delayed
from src.utils.reading_files import read_csv_to_dataframe, read_pkl

# Define the allowed categories, sentiment, and political leanings
categories = [
    'world', 'us', 'politics', 'crime', 'finance', 'scienceandtechnology', 'morenews', 'weather',
    'health', 'autos', 'travel', 'foodanddrink', 'lifestyle',
    'baseball', 'basketball', 'football', 'moresports', 'entertainment',
    'movies', 'music', 'tv', 'video'
]
sentiments = ['negative', 'neutral', 'positive']
politics = ['left', 'center', 'right']

class newsRec_train:

    def __init__(self, hparams, news_file, behaviors_file, seed=None,
    ):
        self.hparams = hparams
        self.news_df= read_csv_to_dataframe(news_file)
        self.behaviors_df= read_csv_to_dataframe(behaviors_file)

        self.news_df = self.news_preprocessing(self.news_df)
        # self.behaviors_df= self.behaviors_mind_preprocessing(self.behaviors_df)

        self.behaviors_df= self.behaviors_preprocessing(self.behaviors_df,  self.hparams)

        self.news_df, self.behaviors_df= self.ids2index(self.news_df, self.behaviors_df, self.hparams)
        # Write the DataFrame to a CSV file


        # news_dict having nidx as keys and value also dict having list of words, list of tokens
        self.news_dict= self.news_dict_prep(self.news_df, self.hparams)

        # # prep input for training (for model)
        self.behaviors_dict = self.behaviors_dict_prep(self.behaviors_df, self.news_dict, self.hparams)
        self.input, self.labels = self.input_prep(self.behaviors_dict)

        # prep input for testing (for scorer)
        #return df [uid, bubble, eval_history(string of words), candidate(explode), label(explode)]
        #also return nparray() same sizes of uid_arr, history_arr, label_arr
        '''
        just comment for mind data training
        '''
        self.behaviors_df_inf, self.arr_uid, self.arr_history, self.arr_candidate, self.arr_label = self.behaviors_prep_inference(self.behaviors_df, self.news_dict, self.hparams)

        # self.uid_one, self.bubble_one, self.input_one, self.labels_one = self.input_prep_one(self.behaviors_dict_one)

        
        # Example word_tokenize function
    def word_tokenize(self, sent):
        # Split sentence into word list using regex.
        pat = re.compile(r"[\w]+|[.,!?;|]")
        if isinstance(sent, str):
            return pat.findall(sent.lower())
        else:
            return []

    def news_preprocessing(self, df):
        #take proper column
        df = df.copy()
        df = df.dropna(subset=['nid', 'category', 'title'])
        return df

    def behaviors_mind_preprocessing(self, df):
        df= df[[1,3,4]]
        df = df.dropna()
        df.rename(columns={1: 'uid', 3: 'history', 4: 'imp'}, inplace=True)

        # Function to extract and clean words
        def extract_words(column, suffix):
            words = column.split()  # Split the string into words
            filtered_words = [word[:-2] for word in words if word.endswith(suffix)]
            return ' '.join(filtered_words)

        # Apply the function to create 'pos' and 'neg' columns
        df['pos'] = df['imp'].apply(lambda x: extract_words(x, '-1'))
        df['neg'] = df['imp'].apply(lambda x: extract_words(x, '-0'))
        df= df[['uid', 'history', 'pos', 'neg']]
        df = df.dropna()

        return df

    def behaviors_preprocessing(self, df, hparams):
        # Each row has one pos
        # Function to expand the DataFrame
        expanded_df = df.explode('pos').reset_index(drop=True)
        # Split the 'pos' column into individual words
        expanded_df = df.assign(pos=df['pos'].str.split())
        # Use the explode method to expand the DataFrame
        expanded_df = expanded_df.explode('pos').reset_index(drop=True)
        df= expanded_df

        #Sample only npratio neg
        def sample_neg_words(neg_str, npratio):
            words = neg_str.split()  # Split the string into words
            if len(words) > npratio:
                sampled_words = ' '.join(random.sample(words, npratio))  # Sample random npratio words
            elif len(words) < npratio:
                sampled_words= words + (['None'] * (npratio - len(words)))
                sampled_words= ' '.join(sampled_words)
            else:
                sampled_words = neg_str  # Keep as it is if less than npratio words
            return sampled_words

        # Apply the function to the neg column
        df['neg'] = df['neg'].apply(lambda x: sample_neg_words(x, hparams.npratio))

        #take only last his_size of the eval_history and padded with none if less than his_size
        # Function to keep only the last his_size words in the eval_history column and pad with None if needed
        def trim_and_pad_history(history_str, his_size):
            words = history_str.split()  # Split the string into words
            if len(words) > his_size:
                # Keep only the last his_size words
                trimmed_history = words[-his_size:]
            else:
                # Pad with None if less than his_size words
                trimmed_history = (['None'] * (his_size - len(words))) + words
            return ' '.join(trimmed_history)

        # Apply the function to the eval_history column
        df['history'] = df['history'].apply(lambda x: trim_and_pad_history(x, hparams.his_size))
        return df

    def ids2index(self, news_df, behaviors_df, hparams):
        # uid_dict = read_pkl(hparams.userDict_file)
        nid_dict = read_pkl(hparams.newsDict_file)
        nid_dict['None']=0

        #  news_df
        # Convert 'id' column to the corresponding values from the dictionary
        news_df['nid'] = news_df['nid'].map(nid_dict)
        # Function to map words to their indices
        def map_ids_to_indices(text, id_to_index_dict):
            words = text.split()  # Split the string into words
            indices = [id_to_index_dict.get(word, word) for word in words]  # Map each word to its index
            return ' '.join(map(str, indices))  # Join the indices back into a string

        # Apply the function to the DataFrame column
        behaviors_df['history'] = behaviors_df['history'].apply(map_ids_to_indices, id_to_index_dict=nid_dict)
        behaviors_df['pos'] = behaviors_df['pos'].apply(map_ids_to_indices, id_to_index_dict=nid_dict)
        behaviors_df['neg'] = behaviors_df['neg'].apply(map_ids_to_indices, id_to_index_dict=nid_dict)


        return news_df, behaviors_df

    def news_dict_prep(self, df, hparams):
        title_size= hparams.title_size
        word_dict = read_pkl(hparams.wordDict_file)
        word_dict['None']= 0

        # Create the news_dict
        news_dict = {}

        for _, row in df.iterrows():
            nid = row['nid']
            sentence = row['title']

            words = self.word_tokenize(sentence)  # Tokenize the sentence into words
            tokens = [word_dict.get(word, 0) for word in words]  # Map words to their indices, default to 0 if not found

            # Adjust the length of words and tokens to title_size
            if len(words) > title_size:
                words = words[:title_size]
                tokens = tokens[:title_size]
            else:
                padding = title_size - len(words)
                words.extend(['None'] * padding)
                tokens.extend([0] * padding)

            news_dict[str(nid)] = {
                'words': words,
                'tokens': tokens
            }
            news_dict['0'] = {
                'words': ['None'] * title_size,
                'tokens': [0] * title_size,
            }

        return news_dict

    def behaviors_dict_prep(self, df, news_dict, hparams):
        behaviors_dict= {}
            # Optimized approach
        def process_row(row, news_dict):
            history = row['history'].split()
            tokens_history = [news_dict[news]['tokens'] for news in history]

            candidate = row['pos'].split() + row['neg'].split()
            label = [1] + [0] * (len(candidate) - 1)

            #Shuffiling
            # Combine the lists into a list of tuples
            combined = list(zip(candidate, label))
            # Shuffle the combined list
            random.shuffle(combined)
            # Unzip the shuffled list back into two lists
            candidate, label = zip(*combined)
            candidate= list(candidate)
            label = list(label)

            tokens_candidate = [news_dict[news]['tokens'] for news in candidate]

            return {'history': tokens_history, 'candidate': tokens_candidate , 'label': label}

        # Apply the optimized function to each row
        behaviors_dict = {index: process_row(row, news_dict) for index, row in df.iterrows()}
        return behaviors_dict

    def input_prep(self, behaviors_dict):
        # Initialize empty lists to store combined values
        combined_history = []
        combined_candidate = []
        combined_labels = []

        # Iterate through the outer dictionary
        for key, sub_dict in behaviors_dict.items():
            combined_history.append(sub_dict['history'])
            combined_candidate.append(sub_dict['candidate'])
            combined_labels.append(sub_dict['label'])

        # Convert lists to numpy arrays
        history_array = np.array(combined_history)
        candidate_array = np.array(combined_candidate)
        labels_array = np.array(combined_labels)

        # Assuming history_array and candidate_array are already defined
        return [history_array, candidate_array], labels_array


    def behaviors_prep_inference(self, df_origin, news_dict, hparams):
        behaviors_dict= {}
            # Optimized approach
        def process_row(row, news_dict):
            history = row['history'].split()
            candidate = row['candidates'].split()
            label = row['labels']
            uid= row['uid']

            tokens_history = [news_dict[news]['tokens'] for news in history]
            tokens_candidate = [news_dict[news]['tokens']  for news in candidate]

            return {'uid': uid, 'history': tokens_history, 'candidate': tokens_candidate , 'label': label}

        def arr_prep(behaviors_dict):
            # Initialize empty lists to store combined values
            combined_uid = []
            combined_history = []
            combined_candidate = []
            combined_labels = []

            # Iterate through the outer dictionary
            for key, sub_dict in behaviors_dict.items():
                combined_uid.append(sub_dict['uid'])
                combined_history.append(sub_dict['history'])
                combined_candidate.append(sub_dict['candidate'])
                combined_labels.append(sub_dict['label'])

            # Convert lists to numpy arrays
            uid_array = np.array(combined_uid).reshape(-1, 1)
            history_array = np.array(combined_history)
            candidate_array = np.array(combined_candidate)
            labels_array = np.array(combined_labels).reshape(-1, 1)

            # Assuming history_array and candidate_array are already defined
            return uid_array, history_array, candidate_array, labels_array

        # expand df with candidte pos and neg
        df=df_origin.copy()
        df['candidates'] = df['pos'] + ' ' + df['neg']
        df['labels'] = df['candidates'].apply(lambda x: '1 ' + '0 ' * (len(x.split()) - 1))

        df=df[['uid', 'bubble', 'history', 'candidates', 'labels' ]]
        df['candidates'] = df['candidates'].str.split()
        df['labels'] = df['labels'].str.split().apply(lambda x: [int(i) for i in x])
        df = df.apply(pd.Series.explode)
        df.reset_index(drop=True, inplace=True)

        # Apply the optimized function to each row
        behaviors_dict = {index: process_row(row, news_dict) for index, row in df.iterrows()}
        uid_arr, history_arr, candidate_arr, label_arr= arr_prep(behaviors_dict)
        return df, uid_arr, history_arr, candidate_arr, label_arr

    def arr_prep(behaviors_dict):
        # Initialize empty lists to store combined values
        combined_uid = []
        combined_history = []
        combined_candidate = []
        combined_labels = []


        # Iterate through the outer dictionary
        for key, sub_dict in behaviors_dict.items():
            combined_uid.append(sub_dict['history'])
            combined_history.append(sub_dict['history'])
            combined_candidate.append(sub_dict['candidate'])
            combined_labels.append(sub_dict['label'])


        # Convert lists to numpy arrays
        history_array = np.array(combined_history)
        candidate_array = np.array(combined_candidate)
        labels_array = np.array(combined_labels)


        # Assuming history_array and candidate_array are already defined
        input = [history_array, candidate_array]
        return input, labels_array

    def input_prep_one(self, behaviors_dict):
        # Initialize empty lists to store combined values
        combined_history = []
        combined_candidate = []
        combined_labels = []
        combined_uid= []
        combined_bubble=[]

        # Iterate through the outer dictionary
        for key, sub_dict in behaviors_dict.items():
            combined_history.append(sub_dict['history'])
            combined_candidate.append(sub_dict['candidate'])
            combined_labels.append(sub_dict['label'])
            combined_uid.append(sub_dict['uid'])
            combined_bubble.append(sub_dict['bubble'])

        # Convert lists to numpy arrays
        history_array = np.array(combined_history)
        candidate_array = np.array(combined_candidate)
        labels_array = np.array(combined_labels)
        uid_array = np.array(combined_uid)
        bubble_array = np.array(combined_bubble)

        # Assuming history_array and candidate_array are already defined
        input = [history_array, candidate_array]
        return uid_array, bubble_array, input, labels_array


class newsRec_inference:

    def __init__(self, hparams, news_file, news_dict, seed=None):

        self.hparams = hparams
        self.news_df = read_csv_to_dataframe(news_file)
        self.news_dict = news_dict
        print(len(self.news_df))


    def behavior_round(self, behaviors_file):
        self.behaviors_df = read_csv_to_dataframe(behaviors_file)

        #Setup:
        # num_users= 1000
        num_cand_per_type= 50
        seed= 42
        ## just for speeding, then remove to have all users
        # self.behaviors_df = self.behaviors_df.head(num_users)


        self.behaviors_df['history'] = self.behaviors_df['history'].fillna('') + ' ' + self.behaviors_df['pos'].fillna('')


        # Extract ony 20 at max for eval_history, padding the lesser
        self.behaviors_df = self.extract_last_history(self.behaviors_df)
        print("extract_last_history and padded by zero if < his_size")



        diff_news_df = self.get_diff_df()
        # combined_df_vertical = pd.concat(diff_news_df, axis=0, ignore_index=True)

        print('diff_news_df')
        def one_row_news_samples(diff_news_df, n):
            news= []
            for df in diff_news_df:
                news.extend(df.sample(n=min(n, len(df)))['nid'].tolist())
            return news

        start_time = time.time()
        news_list= one_row_news_samples(diff_news_df, num_cand_per_type)


        self.behaviors_df['candidate'] = self.behaviors_df.apply(lambda _: one_row_news_samples(diff_news_df, num_cand_per_type), axis=1)


        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        print(elapsed_time)
        print("sample_news")



        # Step 3: Explode the 'candidate' column so each news article has its own row
        self.behaviors_df = self.behaviors_df.explode('candidate').reset_index(drop=True)
        print("behaviors_df.explode")


        start_time = time.time()
        #Extract ndarray of uid, eval_history, candidates
        bubble, uids, history, candidate, history_array, candidate_array= self.extract_ndarray(self.news_dict, self.behaviors_df)

        end_time = time.time()
        elapsed_time = (end_time - start_time)/60
        print(elapsed_time)
        return bubble, uids, history, candidate, history_array, candidate_array

    def get_diff_df(self):
        # Ensure columns are present in the DataFrame
        if not {'category', 'sentiment', 'politics'}.issubset(self.news_df.columns):
            raise ValueError("DataFrame must contain 'category', 'sentiment', and 'politics' columns")

        # Define the allowed categories, sentiment, and political leanings
        categories = [
            'world', 'us', 'politics', 'crime', 'finance', 'scienceandtechnology', 'morenews', 'weather',
            'health', 'autos', 'travel', 'foodanddrink', 'lifestyle',
            'baseball', 'basketball', 'football', 'moresports', 'entertainment',
            'movies', 'music', 'tv', 'video'
        ]
        sentiments = ['Negative', 'Neutral', 'Positive']
        politics = ['Left', 'Center', 'Right']

        ctg_snt_df=[]
        ctg_pol_df= []
        for ctg in categories:
            for s in sentiments:
                temp= self.news_df[(self.news_df['category'] == ctg) & (self.news_df['sentiment'] == s)]
                ctg_snt_df.append(temp)
            for p in politics:
                temp= self.news_df[(self.news_df['category'] == ctg) & (self.news_df['politics'] == p)]
                ctg_pol_df.append(temp)
        return ctg_snt_df + ctg_snt_df



    def extract_last_history(self, df):
        # take only last his_size of the eval_history and padded with none if less than his_size
        # Function to keep only the last his_size words in the eval_history column and pad with None if needed
        def trim_and_pad_history(history_str, his_size):
            words = history_str.split()  # Split the string into words
            if len(words) > his_size:
                # Keep only the last his_size words
                trimmed_history = words[-his_size:]
            else:
                # Pad with None if less than his_size words
                trimmed_history = (['0'] * (his_size - len(words))) + words
            return ' '.join(trimmed_history)

        # Apply the function to the eval_history column
        df['history'] = df['history'].apply(lambda x: trim_and_pad_history(x, self.hparams.his_size))

        return df

    def sample_neg_words(neg_str, npratio):
        words = neg_str.split()  # Split the string into words
        if len(words) > npratio:
            sampled_words = ' '.join(random.sample(words, npratio))  # Sample random npratio words
        elif len(words) < npratio:
            sampled_words = words + (['None'] * (npratio - len(words)))
            sampled_words = ' '.join(sampled_words)
        else:
            sampled_words = neg_str  # Keep as it is if less than npratio words
        return sampled_words

    def ids2index(self, news_df, behaviors_df, hparams):
        uid_dict = read_pkl(hparams.userDict_file)
        nid_dict = read_pkl(hparams.newsDict_file)
        nid_dict['None'] = 0

        #  news_df
        # Convert 'id' column to the corresponding values from the dictionary
        news_df['nid'] = news_df['nid'].map(nid_dict)

        #  behaviors_df
        # Convert 'id' column to the corresponding values from the dictionary
        behaviors_df['uid'] = behaviors_df['uid'].map(uid_dict)

        # column: eval_history, candidate, and neg:
        # Vectorized mapping for 'eval_history', 'candidate', and optionally 'neg' in behaviors_df
        for column in ['history', 'candidate', 'neg', 'pos']:
            if column in behaviors_df.columns:
                behaviors_df[column] = behaviors_df[column].fillna('').astype(str)

                # Apply the transformation safely
                behaviors_df[column] = behaviors_df[column].str.split().apply(
                    lambda x: ' '.join(map(str, [nid_dict.get(word, word) for word in x])) if isinstance(x,list) else '')

        return news_df, behaviors_df

    def news_dict_prep(self, df, hparams):
        title_size = hparams.title_size
        word_dict = read_pkl(hparams.wordDict_file)
        word_dict['None'] = 0

        # Create the news_dict
        news_dict = {}

        for _, row in df.iterrows():
            nid = row['nid']
            sentence = row['title']

            words = self.word_tokenize(sentence)  # Tokenize the sentence into words
            tokens = [word_dict.get(word, 0) for word in words]  # Map words to their indices, default to 0 if not found

            # Adjust the length of words and tokens to title_size
            if len(words) > title_size:
                words = words[:title_size]
                tokens = tokens[:title_size]
            else:
                padding = title_size - len(words)
                words.extend(['None'] * padding)
                tokens.extend([0] * padding)

            news_dict[str(nid)] = {
                'words': words,
                'tokens': tokens
            }
            news_dict['0'] = {
                'words': ['None'] * title_size,
                'tokens': [0] * title_size,
            }

        return news_dict

    def extract_ndarray(self, news_dict, behaviors_df):
        def process_row(row, news_dict):
            # Split the 'eval_history' and 'candidate' fields only once
            history = row['history'].split()
            candidate = str(row['candidate']).split()

            # Use list comprehensions to map the tokens in one go
            tokens_history = [news_dict.get(news, {}).get('tokens', []) for news in history]
            tokens_candidate = [news_dict.get(news, {}).get('tokens', []) for news in candidate]

            return {'history': tokens_history, 'candidate': tokens_candidate}

        # Apply the function across the entire DataFrame and convert the result to a dictionary
        behaviors_dict = behaviors_df.apply(lambda row: process_row(row, news_dict), axis=1).to_dict()
        print("process_row")

        # Initialize empty lists to store combined values
        combined_history = []
        combined_candidate = []

        # Iterate through the outer dictionary
        for key, sub_dict in behaviors_dict.items():
            combined_history.append(sub_dict['history'])
            combined_candidate.append(sub_dict['candidate'])

        # Convert lists to numpy arrays
        uids = np.array(behaviors_df['uid'].tolist()).reshape(-1, 1)
        bubble = np.array(behaviors_df['bubble'].tolist()).reshape(-1, 1)
        history = behaviors_df['history'].apply(lambda x: x.split())
        history = np.array(history.tolist())
        history = np.expand_dims(history, axis=-1)
        candidat = np.array(behaviors_df['candidate'].tolist()).reshape(-1, 1)
        history_array = np.array(combined_history)
        candidate_array = np.array(combined_candidate)

        # Assuming history_array and candidate_array are already defined
        return bubble, uids, history, candidat, history_array, candidate_array


    def word_tokenize(self, sent):
        # Split sentence into word list using regex.
        pat = re.compile(r"[\w]+|[.,!?;|]")
        if isinstance(sent, str):
            return pat.findall(sent.lower())
        else:
            return []

    def news_preprocessing(self, df):
        # take proper column
        df = df.copy()
        df = df.dropna(subset=['nid', 'category', 'title'])
        return df



    def behaviors_preprocessing(self, df, hparams):

        # Sample only npratio neg
        def sample_neg_words(neg_str, npratio):
            words = neg_str.split()  # Split the string into words
            if len(words) > npratio:
                sampled_words = ' '.join(random.sample(words, npratio))  # Sample random npratio words
            elif len(words) < npratio:
                sampled_words = words + (['None'] * (npratio - len(words)))
                sampled_words = ' '.join(sampled_words)
            else:
                sampled_words = neg_str  # Keep as it is if less than npratio words
            return sampled_words

        # Apply the function to the neg column
        df['neg'] = df['neg'].apply(lambda x: sample_neg_words(x, hparams.npratio))

        # take only last his_size of the eval_history and padded with none if less than his_size
        # Function to keep only the last his_size words in the eval_history column and pad with None if needed
        def trim_and_pad_history(history_str, his_size):
            words = history_str.split()  # Split the string into words
            if len(words) > his_size:
                # Keep only the last his_size words
                trimmed_history = words[-his_size:]
            else:
                # Pad with None if less than his_size words
                trimmed_history = (['None'] * (his_size - len(words))) + words
            return ' '.join(trimmed_history)

        # Apply the function to the eval_history column
        df['history'] = df['history'].apply(lambda x: trim_and_pad_history(x, hparams.his_size))

        return df





    def behaviors_dict_prep(self, df, news_dict, hparams):
        behaviors_dict = {}

        # Optimized approach
        def process_row(row, news_dict):
            history = row['history'].split()
            tokens_history = [news_dict[news]['tokens'] for news in history]

            candidate = row['pos'].split() + row['neg'].split()
            label = [1] + [0] * (len(candidate) - 1)

            # Shuffiling
            # Combine the lists into a list of tuples
            combined = list(zip(candidate, label))
            # Shuffle the combined list
            random.shuffle(combined)
            # Unzip the shuffled list back into two lists
            candidate, label = zip(*combined)
            candidate = list(candidate)
            label = list(label)

            tokens_candidate = [news_dict[news]['tokens'] for news in candidate]

            return {'history': tokens_history, 'candidate': tokens_candidate, 'label': label}

        # Apply the optimized function to each row
        behaviors_dict = {index: process_row(row, news_dict) for index, row in df.iterrows()}
        return behaviors_dict

    def behaviors_dict_prep_one(self, df_origin, news_dict, hparams):
        behaviors_dict = {}

        # Optimized approach
        def process_row(row, news_dict):
            history = row['history'].split()
            candidate = row['candidates'].split()
            label = row['labels']

            tokens_history = [news_dict[news]['tokens'] for news in history]
            tokens_candidate = [news_dict[news]['tokens'] for news in candidate]

            return {'history': tokens_history, 'candidate': tokens_candidate, 'label': label}

        # expand df with candidte pos and neg
        df = df_origin.copy()
        df['candidates'] = df['pos'] + ' ' + df['neg']
        df['labels'] = df['candidates'].apply(lambda x: '1 ' + '0 ' * (len(x.split()) - 1))

        df = df[['uid', 'history', 'candidates', 'labels']]
        df['candidates'] = df['candidates'].str.split()
        df['labels'] = df['labels'].str.split().apply(lambda x: [int(i) for i in x])
        df = df.apply(pd.Series.explode)
        df.reset_index(drop=True, inplace=True)

        # Apply the optimized function to each row
        behaviors_dict = {index: process_row(row, news_dict) for index, row in df.iterrows()}
        return behaviors_dict

    def input_prep(self, behaviors_dict):
        # Initialize empty lists to store combined values
        combined_history = []
        combined_candidate = []
        combined_labels = []

        # Iterate through the outer dictionary
        for key, sub_dict in behaviors_dict.items():
            combined_history.append(sub_dict['history'])
            combined_candidate.append(sub_dict['candidate'])
            combined_labels.append(sub_dict['label'])

        # Convert lists to numpy arrays
        history_array = np.array(combined_history)
        candidate_array = np.array(combined_candidate)
        labels_array = np.array(combined_labels)

        # Assuming history_array and candidate_array are already defined
        input = [history_array, candidate_array]
        return input, labels_array



    
