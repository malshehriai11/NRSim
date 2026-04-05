import numpy as np
import time
from collections import defaultdict


from src.utils.reading_files import *
from src.utils.newsrec_utils import prepare_hparams
from src.models.nrms import NRMSModel

from src.data import newsRec_inference
from src.simulation.attributes import get_user_cand_attribute

from src.simulation.utils import news_dict_prep, rank_and_reorder, extract_topk_candidates, interaction_model, prep_behavior_df

seed = 42


news_file = "../../data/news.csv"
newsDict_file = "../../data/nid2index.pkl"
yaml_file = "../utils/nrms.yaml"
wordEmb_file = "../../data/hparam/embedding.npy"
wordDict_file = "../../data/hparam/word_dict.pkl"

categories = [
    'world', 'us', 'politics', 'crime', 'finance', 'scienceandtechnology', "morenews", 'weather',
    'health', 'autos', 'travel', 'foodanddrink', 'lifestyle',
    'baseball', 'basketball', 'football', 'moresports', 'entertainment',
    'movies', 'music', 'tv', 'video'
]


def prep_evaluate_rec2_from_df(df, uid_col, label_col, pred_col):
    """
    Groups labels and predictions by user IDs from a DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - uid_col (str): Name of the column representing user IDs.
    - label_col (str): Name of the column representing labels.
    - pred_col (str): Name of the column representing predictions.

    Returns:
    - grouped_labels (list of np.ndarray): Labels grouped by user IDs.
    - grouped_preds (list of np.ndarray): Predictions grouped by user IDs.
    """
    grouped_labels = defaultdict(list)
    grouped_preds = defaultdict(list)

    # Iterate over DataFrame rows and group by uid
    for _, row in df.iterrows():
        uid = row[uid_col]
        label = row[label_col]
        pred = row[pred_col]

        grouped_labels[uid].append(label)
        grouped_preds[uid].append(pred)

    # Convert lists to np.array
    grouped_labels = [np.array(grouped_labels[uid]) for uid in grouped_labels]
    grouped_preds = [np.array(grouped_preds[uid]) for uid in grouped_preds]

    return grouped_labels, grouped_preds


if __name__ == "__main__":

    train= 't1'


    userDict_file = "../../data/uid2index.pkl"
    userAttrib_file = "../../data/user_attrib_origin.pkl"
    # Model ws
    path_ws = "../../checkpoints/"+train+"/"
    checkpoint = 'ep_10.weights.h5'

    if train=='t1':
        # train 1
        file_path= '../../data/rounds/t1/'
        inputs=['behaviors.csv', 'round_1_behavior_df.csv', 'round_2_behavior_df.csv', 'round_3_behavior_df.csv', 'round_4_behavior_df.csv', 'round_5_behavior_df.csv', 'round_6_behavior_df.csv', 'round_7_behavior_df.csv', 'round_8_behavior_df.csv', 'round_9_behavior_df.csv']

        outputs= ['round_1', 'round_2','round_3','round_4','round_5','round_6','round_7','round_8','round_9','round_10']



    else:
    # train 2 or greater
        file_path= '../data/rounds/'+train+'/'
        #First file from previous training - round 10
        first_file= '../data/rounds/t1/round_10_behavior_df.csv'

        inputs=[first_file, 'round_1_behavior_df.csv', 'round_2_behavior_df.csv', 'round_3_behavior_df.csv', 'round_4_behavior_df.csv', 'round_5_behavior_df.csv', 'round_6_behavior_df.csv','round_7_behavior_df.csv', 'round_8_behavior_df.csv', 'round_9_behavior_df.csv']

        outputs= ['round_1','round_2','round_3','round_4','round_5','round_6','round_7','round_8','round_9','round_10']



    news_df = read_csv_to_dataframe(news_file)
    hparams = prepare_hparams(yaml_file,
                              wordEmb_file=wordEmb_file,
                              wordDict_file=wordDict_file,
                              userDict_file=userDict_file,
                              newsDict_file=newsDict_file,
                              batch_size=32,
                              epochs=5,
                              show_step=10)
    # #     # #
    print('hparams')
    model = NRMSModel(hparams, seed=seed)

    model.model.load_weights(path_ws + 'model/' + checkpoint)
    model.scorer.load_weights(path_ws + 'scorer/' + checkpoint)
    print('Model loaded')
    news_dict= news_dict_prep(news_file, hparams)
    data= newsRec_inference(hparams, news_file, news_dict)

    for i in range(len(inputs)):
        very_start_time=time.time()

        #for training 2 and greater
        # if i == 0:
        #     behaviors_file= inputs[i]
        # else:
        #     behaviors_file= file_path+ inputs[i]

        behaviors_file= file_path+ inputs[i]
        bubble, uid, history, candidate, history_arr, candidate_arr= data.behavior_round(behaviors_file)

        print('Prediction:')
        pred= model.predict_new([history_arr, candidate_arr], batch_size=128)

        start_time = time.time()
        sorted_uid, sorted_bubble, sorted_history, sorted_candidate, sorted_history_arr, sorted_candidate_arr, sorted_pred = rank_and_reorder(uid, bubble, history, candidate, history_arr, candidate_arr, pred)
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        print(elapsed_time)

        # # # # Extract top_k
        start_time = time.time()
        topk_uid, topk_bubble,  topk_history, topk_candidate, topk_history_arr, topk_candidate_arr, topk_pred= extract_topk_candidates(20, sorted_uid, sorted_bubble, sorted_history, sorted_candidate, sorted_history_arr, sorted_candidate_arr, sorted_pred)
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        print(elapsed_time)


        '''
        Interaction models,
        '''
        attribute_list= get_user_cand_attribute(news_file, categories, topk_bubble, topk_uid, topk_history, topk_candidate, dynamic_user_update= True)
        print('get_user_cand_attribute --done')

        topk_interaction_result= interaction_model(attribute_list)
        print('interaction_model --done')


        '''
        obtained round dfs, and save it
        '''

        round_df, round_behavior_df = prep_behavior_df(topk_bubble, topk_uid, topk_history, topk_candidate, topk_pred, topk_interaction_result)

        round_df.to_csv(file_path+ outputs[i]+'.csv', index=False)
        round_behavior_df.to_csv(file_path + outputs[i] + '_behavior_df.csv', index=False)

        end_time = time.time()
        elapsed_time = (end_time - very_start_time) / 60
        print('total time= '+ str(elapsed_time) + ' mins')

