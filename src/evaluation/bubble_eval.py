import numpy as np
import pickle

categ = [
        'world', 'us', 'politics', 'crime', 'finance', 'scienceandtechnology', "morenews", 'weather',
        'health', 'autos', 'travel', 'foodanddrink', 'lifestyle',
        'baseball', 'basketball', 'football', 'moresports', 'entertainment',
        'movies', 'music', 'tv', 'video'
    ]


# Load dictionary from pickle file
with open("../../data/user_attrib_origin.pkl", 'rb') as f:
    user_attrib_origin = pickle.load(f)

lamda= 0.2

def compute_entropy(array):
    array = array[array > 0]  # Remove zero values to avoid log(0)
    if len(array) <= 1:
        return 0.0  # Entropy is zero if there is only one category
    entropy = -np.sum(array * np.log2(array))
    return entropy

def categories_distribution_info(category_string, categories= categ, dynamic_user_update=False, user=None):

    # # Split the input string into a list of categories
    category_list = category_string.split()


    # Count occurrences of each category
    category_count = {cat: category_list.count(cat) for cat in categories}

    # Calculate total count for normalization
    total_count = sum(category_count.values())

    # Calculate the probability distribution
    if dynamic_user_update:
        u_origin= user_attrib_origin[user]['user_prob_dist']
        if total_count > 0:
            prob_distribution = u_origin + lamda * np.array([category_count[cat] / total_count for cat in categories])
            prob_distribution = prob_distribution / np.sum(prob_distribution)
        else:
            prob_distribution = u_origin + lamda * np.zeros(len(categories))
            prob_distribution = prob_distribution / np.sum(prob_distribution)
    else:
        if total_count > 0:
            prob_distribution = np.array([category_count[cat] / total_count for cat in categories])
        else:
            prob_distribution = np.zeros(len(categories))

    # Find the maximum probability and corresponding category
    max_prob_index = np.argmax(prob_distribution)
    max_prob = prob_distribution[max_prob_index]
    max_category = categories[max_prob_index]

    # Compute entropy of the probability distribution
    entropy = compute_entropy(prob_distribution)

    # Calculate the number of unique categories with non-zero probability
    num_categories = np.sum(prob_distribution > 0)

    # Compute the maximum possible entropy
    max_entropy = np.log2(num_categories) if num_categories > 1 else 1  # Ensure max_entropy is 1 for a single category or less

    # Compute normalized entropy
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Create the output dictionary
    result = {
        'categories_prob_dist': prob_distribution,
        'categories_max_prob': max_prob,
        'categories_max_category': max_category,
        'categories_normalized_entropy': normalized_entropy
    }

    return result

def sentiment_distribution_info(sentiment_string, dynamic_user_update=False, user=None):
    sentiment_string = sentiment_string.lower()
    # Define the sentiment mapping
    sentiment_map = {"negative": -1, "neutral": 0, "positive": 1, "mixed": 0}

    # Split the input string into words and map them to integer values
    sentiments = sentiment_string.split()
    mapped_values = [sentiment_map.get(sentiment.strip(), 0) for sentiment in sentiments]

    # Calculate sentiment counts
    sentiment_count = {key: sentiments.count(key) for key in sentiment_map.keys()}
    del sentiment_count['mixed']

    # Total count for normalization
    total_count = sum(sentiment_count.values())

    # Calculate the probability distribution
    if dynamic_user_update:
        u_origin= user_attrib_origin[user]['user_prob_snt']
        if total_count > 0:
            prob_distribution = u_origin + lamda * np.array([sentiment_count[sentiment] / total_count for sentiment in sentiment_count.keys()])
            prob_distribution = prob_distribution / np.sum(prob_distribution)

        else:
            prob_distribution = u_origin + lamda * np.zeros(len(sentiment_map))
            prob_distribution = prob_distribution / np.sum(prob_distribution)

    else:
        if total_count > 0:
            prob_distribution = np.array(
                [sentiment_count[sentiment] / total_count for sentiment in sentiment_count.keys()])
        else:
            prob_distribution = np.zeros(len(sentiment_map))

    # Find the sentiment with the highest probability
    max_prob_index = np.argmax(prob_distribution)
    max_prob = prob_distribution[max_prob_index]
    max_sentiment = list(sentiment_map.keys())[max_prob_index]


    # Compute entropy of the probability distribution
    entropy = compute_entropy(prob_distribution)

    # Calculate the number of unique sentiments with non-zero probability
    num_sentiments = np.sum(prob_distribution > 0)

    # Compute the maximum possible entropy
    max_entropy = np.log2(num_sentiments) if num_sentiments > 1 else 1

    # Compute normalized entropy
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Compute the average sentiment value
    if dynamic_user_update:
        u_origin = user_attrib_origin[user]['user_avg_snt']
        average_sentiment = u_origin + lamda * sum(mapped_values) / len(mapped_values)  if mapped_values else 0
        average_sentiment = average_sentiment / max(1, abs(average_sentiment))
    else:
        average_sentiment = sum(mapped_values) / len(mapped_values) if mapped_values else 0


    # Create the output dictionary
    result = {
        'sentiment_prob_dist': prob_distribution,
        'sentiment_max_prob': max_prob,
        'sentiment_max_sentiment': max_sentiment,
        'sentiment_normalized_entropy': normalized_entropy,
        'average_sentiment': average_sentiment,
        'final_sentiment': 'negative' if average_sentiment <= -0.5 else 'positive' if average_sentiment >= 0.5 else 'neutral'
    }

    return result

def politics_distribution_info(politics_string, dynamic_user_update=False, user=None):
    politics_string = politics_string.lower()


    # Define the political leaning mapping
    politics_map = {"left": -1, "center": 0, "right": 1, "mixed": 0, "center-left": 0}

    # Split the input string into words and map them to integer values
    politics = politics_string.split()
    mapped_values = [politics_map.get(politic.strip(), 0) for politic in politics]

    # Count occurrences of each political leaning
    politics_count = {key: politics.count(key) for key in politics_map.keys()}
    del politics_count['mixed']
    del politics_count['center-left']


    # Total count for normalization
    total_count = sum(politics_count.values())

    # Calculate the probability distribution
    if dynamic_user_update:
        u_origin = user_attrib_origin[user]['user_prob_pol']
        if total_count > 0:
            prob_distribution = u_origin + lamda * np.array([politics_count[politic] / total_count for politic in politics_count.keys()])
            prob_distribution = prob_distribution / np.sum(prob_distribution)

        else:
            prob_distribution = u_origin + lamda * np.zeros(len(politics_map))
            prob_distribution = prob_distribution / np.sum(prob_distribution)

    else:
        if total_count > 0:
            prob_distribution = np.array([politics_count[politic] / total_count for politic in politics_count.keys()])
        else:
            prob_distribution = np.zeros(len(politics_map))


    # Find the political leaning with the highest probability
    max_prob_index = np.argmax(prob_distribution)
    max_prob = prob_distribution[max_prob_index]
    max_politic = list(politics_map.keys())[max_prob_index]


    # Compute entropy of the probability distribution
    entropy = compute_entropy(prob_distribution)

    # Calculate the number of unique political leanings with non-zero probability
    num_politics = np.sum(prob_distribution > 0)

    # Compute the maximum possible entropy
    max_entropy = np.log2(num_politics) if num_politics > 1 else 1

    # Compute normalized entropy
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Compute the average political leaning value
    if dynamic_user_update:
        u_origin = user_attrib_origin[user]['user_avg_pol']
        average_politics = u_origin + lamda * sum(mapped_values) / len(mapped_values) if mapped_values else 0
        average_politics = average_politics / max(1, abs(average_politics))

    else:
        average_politics = sum(mapped_values) / len(mapped_values)  if mapped_values else 0


    # Create the output dictionary
    result = {
        'politics_prob_dist': prob_distribution,
        'politics_max_prob': max_prob,
        'politics_max_leaning': max_politic,
        'politics_normalized_entropy': normalized_entropy,
        'average_politics': average_politics,
        'final_politics': 'left' if average_politics <= -0.5 else 'right' if average_politics >= 0.5 else 'center'
    }


    return result






if __name__ == "__main__":
    # Example usage
    categories = [
        'world', 'us', 'politics', 'crime', 'finance', 'scienceandtechnology', "morenews", 'weather',
        'health', 'autos', 'travel', 'foodanddrink', 'lifestyle',
        'baseball', 'basketball', 'football', 'moresports', 'entertainment',
        'movies', 'music', 'tv', 'video'
    ]

    # Example input string
    category_string = 'music travel health football foodanddrink baseball crime baseball basketball world finance football music us health music health music tv crime'
    # Apply the function
    result = categories_distribution_info(categories, category_string)
    print(result)

    category_string = 'music'
    # Apply the function
    result = categories_distribution_info(categories, category_string)
    print(result)

    #------------------------------------

    # Example input strings
    sentiment_string = "negative negative positive negative positive negative negative negative positive negative negative neutral negative positive positive positive negative positive negative negative"
    politics_string = "center center center center center center center center center left center center center center center center left center center center"

    # Calculate the average sentiment and political leaning
    average_sentiment = sentiment_distribution_info(sentiment_string)
    average_politics = politics_distribution_info(politics_string)

    print(f"Average Sentiment: {average_sentiment}")
    print(f"Average Politics: {average_politics}")

    sentiment_string = "negative"
    politics_string = "center"

    # Calculate the average sentiment and political leaning
    average_sentiment = sentiment_distribution_info(sentiment_string)
    average_politics = politics_distribution_info(politics_string)

    print(f"Average Sentiment: {average_sentiment}")
    print(f"Average Politics: {average_politics}")

