
import numpy as np
import pandas as pd
import random
import re
from joblib import Parallel, delayed
from src.utils.reading_files import read_csv_to_dataframe, read_pkl
from src.utils.newsrec_utils import prepare_hparams


seed = 42


news_file = "data/news.csv"
news_df = read_csv_to_dataframe(news_file)


def ids2index(df, behaviors_df, hparams):
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
    for column in ['eval_history', 'pos', 'neg']:
        if column in behaviors_df.columns:
            # Split the text into lists, map the IDs, and join back into strings
            behaviors_df[column] = behaviors_df[column].str.split().apply(
                lambda x: ' '.join(map(str, [nid_dict.get(word, word) for word in x])))

    return news_df, behaviors_df

def word_tokenize(sent):
        # Split sentence into word list using regex.
        pat = re.compile(r"[\w]+|[.,!?;|]")
        if isinstance(sent, str):
            return pat.findall(sent.lower())
        else:
            return []

def news_dict_prep(news_file, hparams):
        df = read_csv_to_dataframe(news_file)

        title_size = hparams.title_size
        word_dict = read_pkl(hparams.wordDict_file)
        word_dict['None'] = 0

        # Create the news_dict
        news_dict = {}

        for _, row in df.iterrows():
            nid = row['nid']
            sentence = row['title']

            words = word_tokenize(sentence)  # Tokenize the sentence into words
            tokens = [word_dict.get(word, 0) for word in words]# Map words to their indices, default to 0 if not found

            # Adjust the length of words and tokens to title_size
            if len(words) > title_size:
                words = words[:title_size]
                tokens = tokens[:title_size]
            else:
                padding = title_size - len(words)
                words.extend(['None'] * padding)
                tokens.extend([0] * padding)

            tokens= np.array(tokens)

            news_dict[str(nid)] = {
                'words': words,
                'tokens': tokens
            }
            news_dict['0'] = {
                'words': ['None'] * title_size,
                'tokens': np.array([0] * title_size)
            }

        return news_dict

def rank_and_reorder(user_ids, *arrays_to_sort):
    """
    Reorders arrays based on the descending order of the values in the last array,
    grouped by unique user IDs.

    Parameters:
    - user_ids (np.ndarray): Array of user IDs, potentially with duplicates, shape (n, 1).
    - *arrays_to_sort (np.ndarray): Arrays to reorder based on the values in the last array,
                                    shapes should match or be broadcastable with user_ids.

    Returns:
    - A tuple of np.ndarrays: Reordered arrays including the reordered user_ids.
    """
    # Ensure all input arrays have the same first dimension
    n = len(user_ids)
    if not all(arr.shape[0] == n for arr in arrays_to_sort):
        raise ValueError("All input arrays must have the same number of rows (first dimension).")

    # The array that determines the sorting order is the last one (prediction)
    sorting_array = arrays_to_sort[-1].reshape(n, -1)  # Flatten the last two dimensions

    # Get unique user IDs and their sorted indices
    unique_users, inverse_indices = np.unique(user_ids, return_inverse=True)

    # Prepare a list to store the sorted results
    sorted_results = [np.empty_like(arr) for arr in (user_ids, *arrays_to_sort)]

    # Sort each group based on the descending order of the sorting_array
    for user in range(len(unique_users)):
        if user % 100 == 0:  # Check if the user is 100, 200, ...
            print(user)  # Get indices for the current user

        user_mask = inverse_indices == user

        # Sort the sorting_array for the current user in descending order
        sort_indices = np.argsort(-sorting_array[user_mask], axis=0).squeeze()

        # Apply the sorting to all relevant arrays
        for i, arr in enumerate((user_ids, *arrays_to_sort)):
            if arr.ndim > 2:  # For higher dimensional arrays, we need to index along the first dimension only
                sorted_results[i][user_mask] = arr[user_mask][sort_indices, :]
            else:
                sorted_results[i][user_mask] = arr[user_mask][sort_indices]

    return tuple(sorted_results)

def extract_topk_candidates(k, *arrays_to_sort):
    # Initialize an array to track the number of top-k elements per user
    user_ids= arrays_to_sort[0]
    # user_counts = np.bincount(user_ids.flatten())
    user_counts = np.bincount(user_ids.flatten().astype(int))

    # Calculate the total number of elements in the top-k selection
    total_topk = np.sum(np.minimum(user_counts, k))

    # Preallocate array for the top-k users
    topk_user_ids = np.empty((total_topk, user_ids.shape[1]), dtype=user_ids.dtype)

    # Preallocate arrays for the other sorted arrays
    sorted_arrays = [np.empty((total_topk,) + arr.shape[1:], dtype=arr.dtype) for arr in arrays_to_sort]

    current_index = 0

    for user_id in np.unique(user_ids):
        if user_id % 100 == 0:  # Check if the user is 100, 200, ...
            print(user_id)        # Find the indices corresponding to the current user
        user_indices = np.where(user_ids == user_id)[0]

        # Sort the user indices by the prediction values (descending)
        predictions = arrays_to_sort[-1]
        sorted_indices = user_indices[np.argsort(predictions[user_indices].flatten())[::-1]]

        # Select the top-k indices
        topk_indices = sorted_indices[:k]

        # Determine how many rows to add to the final arrays
        num_topk = len(topk_indices)

        # Assign the top-k elements to the preallocated arrays
        topk_user_ids[current_index:current_index + num_topk] = user_ids[topk_indices]

        for i, arr in enumerate(sorted_arrays):
            arr[current_index:current_index + num_topk] = arrays_to_sort[i][topk_indices]

        current_index += num_topk

    # Replace the original arrays with the sorted ones
    for i in range(len(arrays_to_sort)):
        print(i)
        arrays_to_sort[i].resize(sorted_arrays[i].shape, refcheck=False)
        arrays_to_sort[i][:] = sorted_arrays[i]

    return tuple(sorted_arrays)

def interaction_model(attribute_list):
    # Use list comprehension to compute interactions efficiently
    pred = [compute_interaction(attribute_list[i]) for i in range(len(attribute_list))]
    # Convert list to ndarray and reshape to (n, 1)
    # pred = np.array(pred).reshape(-1, 1)
    pred = np.array([item[0] if isinstance(item, (list, np.ndarray)) else item for item in pred])
    pred = pred.reshape(-1, 1)

    return pred

def compute_interaction(one_interaction_dict):

    # Dynamic weight calculation based on user preferences
    category_weights = 1 + 1 * (1 - one_interaction_dict['user_entropy'])  # Higher weight for lower entropy
    sentiment_weights = 1 + 1 * abs(
        one_interaction_dict['user_avg_snt'])  # Higher weight for stronger sentiment preferences
    politics_weights = 1 + 1 * abs(
        one_interaction_dict['user_avg_pol'])  # Higher weight for stronger political preferences


    user_prob_dist = one_interaction_dict['user_prob_dist'].reshape(1, -1)
    user_prob_snt = one_interaction_dict['user_prob_snt'].reshape(1, -1)
    user_prob_pol = one_interaction_dict['user_prob_pol'].reshape(1, -1)

    cand_one_hot = one_interaction_dict['cand_prob_dist'].reshape(1, -1)
    cand_sentiment_one_hot = one_interaction_dict['cand_prob_snt'].reshape(1, -1)
    cand_politics_one_hot = one_interaction_dict['cand_prob_pol'].reshape(1, -1)

    if cand_sentiment_one_hot.shape != (1,3) or cand_politics_one_hot.shape != (1,3):
        print(one_interaction_dict)

    # to delete this record
    if any(arr.shape != (1, 3) for arr in
           [user_prob_snt, user_prob_pol, cand_sentiment_one_hot, cand_politics_one_hot]):
        interaction_result = 0
    else:
        # Compute dot products for each user-article pair (element-wise multiplication and sum)
        topic_similarity = np.sum(user_prob_dist * cand_one_hot, axis=1)  # Category similarity
        sentiment_similarity = np.sum(user_prob_snt * cand_sentiment_one_hot, axis=1)  # Sentiment similarity
        political_similarity = np.sum(user_prob_pol * cand_politics_one_hot, axis=1)  # Political similarity

        # Combine the similarities using the dynamic weights
        overall_similarity = (
                (topic_similarity * category_weights +
                 sentiment_similarity * sentiment_weights +
                 political_similarity * politics_weights) /
                (category_weights + sentiment_weights + politics_weights)
        )

        # Set a threshold (e.g., 0.5)
        threshold = 0.3
        # Introduce very small random noise within a tiny range, e.g., [-1e-5, 1e-5]
        small_noise = np.random.uniform(-1e-5, 1e-5)
        interaction_value_noisy = overall_similarity + small_noise
        interaction_value_noisy = overall_similarity

        # Determine if it's an interaction
        interaction = interaction_value_noisy >= threshold
        interaction_result = np.where(interaction, 1, 0)

    return interaction_result

def prep_behavior_df(*arrays):
    topk_bubble= arrays[0]
    topk_uid= arrays[1]
    topk_history = arrays[2]
    topk_candidate = arrays[3]
    topk_pred= arrays[4]
    topk_interaction_result = arrays[5]

    # Flatten the input arrays
    bubbles= topk_bubble.flatten()
    uids = topk_uid.flatten()
    candidates = topk_candidate.flatten()
    pred = topk_pred.flatten()
    interaction_results = topk_interaction_result.flatten()

    # Convert eval_history to a list of strings
    histories = [' '.join(map(str, hist.flatten())) for hist in topk_history]
    # histories = [' '.join(hist) for hist in topk_history]

    # Create a DataFrame to work with
    df = pd.DataFrame(
        {'bubble':bubbles, 'uid': uids, 'history': histories, 'candidates': candidates, 'prediction': pred, 'interaction_result': interaction_results})


    # Use dictionary to aggregate data instead of groupby
    unique_uids = np.unique(uids)
    data = {'bubble': [], 'uid': [], 'history': [], 'pos': [], 'neg': []}

    # Process each unique user
    for uid in unique_uids:
        user_data = df[df['uid'] == uid]
        data['bubble'].append(user_data['bubble'].iloc[0])
        data['uid'].append(uid)
        data['history'].append(user_data['history'].iloc[0])  # First occurrence of eval_history
        data['pos'].append(' '.join(user_data[user_data['interaction_result'] == 1]['candidates'].astype(str)))
        data['neg'].append(' '.join(user_data[user_data['interaction_result'] == 0]['candidates'].astype(str)))

    # Create a new DataFrame from the dictionary
    result_df = pd.DataFrame(data)

    return df, result_df


import cupy as cp

def rank_and_reorder_gpu(user_ids, *arrays_to_sort):
    """
    Reorders arrays based on the descending order of the values in the last array,
    grouped by unique user IDs, running on a GPU using CuPy.

    Parameters:
    - user_ids (cp.ndarray): Array of user IDs, potentially with duplicates, shape (n, 1).
    - *arrays_to_sort (cp.ndarray): Arrays to reorder based on the values in the last array,
                                    shapes should match or be broadcastable with user_ids.

    Returns:
    - A tuple of cp.ndarrays: Reordered arrays including the reordered user_ids.
    """
    # Convert to CuPy arrays
    user_ids = cp.asarray(user_ids)
    arrays_to_sort = [cp.asarray(arr) for arr in arrays_to_sort]

    # The array that determines the sorting order is the last one (prediction)
    sorting_array = arrays_to_sort[-1].reshape(user_ids.shape[0], -1)

    # Get unique user IDs and their sorted indices
    unique_users, inverse_indices = cp.unique(user_ids, return_inverse=True)

    # Prepare a list to store the sorted results
    sorted_results = [cp.empty_like(arr) for arr in (user_ids, *arrays_to_sort)]

    # Sort each group based on the descending order of the sorting_array
    for user in range(len(unique_users)):
        print(user)
        # Get indices for the current user
        user_mask = inverse_indices == user

        # Sort the sorting_array for the current user in descending order
        sort_indices = cp.argsort(-sorting_array[user_mask], axis=0).squeeze()

        # Apply the sorting to all relevant arrays
        for i, arr in enumerate((user_ids, *arrays_to_sort)):
            if arr.ndim > 2:  # For higher-dimensional arrays
                sorted_results[i][user_mask] = arr[user_mask][sort_indices, :]
            else:
                sorted_results[i][user_mask] = arr[user_mask][sort_indices]

    # Convert results back to NumPy if needed
    return tuple(cp.asnumpy(arr) for arr in sorted_results)


def extract_topk_candidates_gpu(k, *arrays_to_sort):
    """
    Extract the top-k candidates per user and run the function on a GPU using CuPy.

    Parameters:
    - k (int): The number of top elements to select per user.
    - *arrays_to_sort (cp.ndarray): Arrays to reorder and extract top-k elements.

    Returns:
    - A tuple of cp.ndarrays: Arrays containing the top-k elements for each user.
    """
    # Extract user IDs from the first array
    user_ids = cp.asarray(arrays_to_sort[0])
    arrays_to_sort = [cp.asarray(arr) for arr in arrays_to_sort]

    # Calculate user counts
    user_counts = cp.bincount(user_ids.flatten().astype(int))

    # Calculate the total number of elements in the top-k selection
    total_topk = cp.sum(cp.minimum(user_counts, k)).get()  # Convert to host for preallocation

    # Preallocate arrays for the top-k elements
    topk_user_ids = cp.empty((total_topk, user_ids.shape[1]), dtype=user_ids.dtype)
    sorted_arrays = [cp.empty((total_topk,) + arr.shape[1:], dtype=arr.dtype) for arr in arrays_to_sort]

    current_index = 0

    # Get unique user IDs
    unique_user_ids = cp.unique(user_ids)

    # Iterate over each unique user
    for user_id in unique_user_ids:
        # Find the indices for the current user
        user_indices = cp.where(user_ids == user_id)[0]

        # Sort the user indices by the prediction values (descending)
        predictions = arrays_to_sort[-1]
        sorted_indices = user_indices[cp.argsort(predictions[user_indices].flatten())[::-1]]

        # Select the top-k indices
        topk_indices = sorted_indices[:k]

        # Determine how many rows to add to the final arrays
        num_topk = len(topk_indices)

        # Assign the top-k elements to the preallocated arrays
        topk_user_ids[current_index:current_index + num_topk] = user_ids[topk_indices]

        for i, arr in enumerate(sorted_arrays):
            arr[current_index:current_index + num_topk] = arrays_to_sort[i][topk_indices]

        current_index += num_topk

    # Convert back to NumPy arrays if needed
    return tuple(cp.asnumpy(arr) for arr in (topk_user_ids, *sorted_arrays))

def predict_gpu(self, input_data, batch_size=512):
    """
    Predict function optimized for better GPU utilization and performance.

    Args:
        input_data: List of two ndarrays [history, candidate].
        batch_size: Batch size for prediction.

    Returns:
        np.ndarray: Predicted values concatenated into a single array.
    """
    # Split input data
    history, candidate = input_data

    # Convert input data to TensorFlow tensors
    history = tf.convert_to_tensor(history, dtype=tf.float32)
    candidate = tf.convert_to_tensor(candidate, dtype=tf.float32)

    # Create TensorFlow dataset for batching and prefetching
    dataset = tf.data.Dataset.from_tensor_slices((history, candidate))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Initialize a list to store predictions
    predictions = []

    # Use tqdm to display progress
    for batch in tqdm(dataset, desc="Batches", unit="batch"):
        # Predict on batch
        batch_predictions = self.scorer.predict_on_batch(batch)
        predictions.append(batch_predictions)

    # Concatenate all batch predictions into a single array
    predictions = np.concatenate(predictions, axis=0)

    return predictions
