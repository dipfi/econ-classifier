'''
This script is part of the Master Thesis of Damian Durrer (SfS ETH Zürich), submission date 16 December 2021.

This script is part of the following project:
- Github: https://github.com/dipfi/econ-classifier
- Euler: /cluster/work/lawecon/Projects/Ash_Durrer/dev/scripts

The data for reproduction can be found on:
- K-drive: https://drive.infomaniak.com/app/share/249519/d8dab04d-7ced-4f3a-a995-1916b3aa03a9
- Euler: /cluster/work/lawecon/Projects/Ash_Durrer/dev/data
--> The relevant config-files for github and the profile settings for Euler are in the "notes" folders


Much of the code here is based on:
https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794
'''

import pandas as pd
import logging
logger = logging.getLogger(__name__)

## for language detection
import langdetect
import multiprocessing as mp

## for preprocessing
import nltk
import re
import os

## for evaluation
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing, feature_selection, metrics
import numpy as np


def hello_world():
    print("hello_world")




def load_data(data_path,
              input_file_name,
              input_file_size,
              input_file_type,
              sample_size):

    logger.info("Function started: load_data")

    if input_file_size == "all":
        input_file_fullname = input_file_name + "." + input_file_type
    else:
        input_file_fullname = input_file_name + "_" + str(input_file_size) + "." + input_file_type

    if input_file_type == "csv":
        logger.info("input_file_type is CSV")
        dtf = pd.read_csv(data_path + "/" + input_file_fullname)
    else:
        logger.info("input_file_type is not CSV")

    if sample_size != "all":
        dtf = dtf.sample(sample_size).copy()

    logger.info("Data loaded")
    return dtf
    logger.info("Function ended: load_data")



def save_data_csv(dtf,
                  data_path,
                  output_file_name,
                  sample_size):
    logger.info("Function started: save_data_csv")

    if sample_size == "all":
        new_sample_file_name = data_path + '/' + output_file_name + '.csv'

    else:
        new_sample_file_name = data_path + '/' + output_file_name + "_" + str(sample_size) + '.csv'

    logging.info("SAVING NEW SAMPLE FILE: " + new_sample_file_name)
    dtf.to_csv(new_sample_file_name, index=False)

    logger.info("Function ended: save_data_csv")


def save_results(data_path,
                 results_file_name,
                 result,
                 dtf_test,
                 trainingset_id,
                 result_id,
                 predicted_prob,
                 y_test_bin):

    logger.info("Function started: save_results")

    results_path = data_path + "/results/" + str(results_file_name) + ".csv"

    try:
        results = pd.read_csv(results_path)
        results = pd.concat([results, result])
        results.to_csv(results_path, index=False)
    except FileNotFoundError:
        results = result
        results.to_csv(results_path, index=False)
    except PermissionError:

        counter = 0
        trying_to_save = True
        results = result

        while trying_to_save and counter < 100:

            try:
                results_path_new = data_path + "/results/" + str(results_file_name) + str(counter) + ".csv"
                results.to_csv(results_path_new, index=False)
                trying_to_save = False
            except PermissionError:
                counter += 1

    try:
        journal = dtf_test["Journal"].tolist()
    except:
        journal = [None for i in range(len(dtf_test))]
    try:
        pubyear = dtf_test["Publication Year"].tolist()
    except:
        pubyear = [None for i in range(len(dtf_test))]
    try:
        WOS_ID = dtf_test["UT (Unique WOS ID)"].tolist()
    except:
        WOS_ID = [None for i in range(len(dtf_test))]

    result_id_list = [result_id for i in range(len(dtf_test))]
    trainingset_id_list = [trainingset_id for i in range(len(dtf_test))]

    pred_prob_df = pd.DataFrame({"probabilities": predicted_prob[:, 1].tolist(),
                                 "true_value": y_test_bin.tolist(),
                                 "journal": journal,
                                 "year": pubyear,
                                 "ID": WOS_ID,
                                 "result_id": result_id_list,
                                 "trainingset_id": trainingset_id_list
                                 })

    pred_prob_df.to_csv(data_path + "/pred_prob/" + str(result_id) + ".csv", index=False)

    logger.info("Function ended: save_results")

    return results, pred_prob_df


def evaluate(classes,
             # y_test,
             y_test_bin,
             # predicted,
             predicted_bin,
             predicted_prob):

    logger.info("Function started: evaluate")

    cm = metrics.confusion_matrix(y_test_bin, predicted_bin)
    logger.info("confusion matrix: " + str(cm))

    try:
        auc = metrics.roc_auc_score(y_test_bin, predicted_prob[:, 1])
    except:
        auc = None

    try:
        precision, recall, threshold = metrics.precision_recall_curve(y_test_bin, predicted_prob[:, 1])
        auc_pr = metrics.auc(recall, precision)
    except ValueError:
        auc_pr = None

    try:
        mse_negative = metrics.mean_squared_error(y_test_bin[y_test_bin == 0], predicted_prob[:, 1][y_test_bin == 0])
    except ValueError:
        mse_negative = None

    try:
        mse_positive = metrics.mean_squared_error(y_test_bin[y_test_bin == 1], predicted_prob[:, 1][y_test_bin == 1])
    except ValueError:
        mse_positive = None

    try:
        mse_average = (mse_negative + mse_positive) / 2
    except TypeError:
        mse_average = None

    try:
        mcc = metrics.matthews_corrcoef(y_test_bin, predicted_bin)
    except ValueError:
        mcc = None

    Negative_Label = classes[0]
    Positive_Label = classes[1]
    Support_Negative = len(y_test_bin[y_test_bin == 0])
    Support_Positive = len(y_test_bin[y_test_bin == 1])
    TN = np.sum([test == 0 and pred == 0 for test, pred in zip((y_test_bin).tolist(), (predicted_bin).tolist())])
    FP = np.sum([test == 0 and pred == 1 for test, pred in zip((y_test_bin).tolist(), (predicted_bin).tolist())])
    FN = np.sum([test == 1 and pred == 0 for test, pred in zip((y_test_bin).tolist(), (predicted_bin).tolist())])
    TP = np.sum([test == 1 and pred == 1 for test, pred in zip((y_test_bin).tolist(), (predicted_bin).tolist())])
    Precision_Negative = TN / (FN + TN)
    Precision_Positive = TP / (FP + TP)
    Recall_Negative = TN / (FP + TN)
    Recall_Positive = TP / (FN + TP)
    F1_Score = 2*((Precision_Positive*Recall_Positive)/(Precision_Positive+Recall_Positive))
    if Support_Negative == 0:
        Label = "1heterodox"
        Recall = Recall_Positive
    elif Support_Positive == 0:
        Label = "0orthodox"
        Recall = Recall_Negative
    else:
        Label = None
        Recall = None
    AUC = auc
    AUC_PR = auc_pr
    MCC = mcc
    MSE_NEGATIVE = mse_negative
    MSE_POSITIVE = mse_positive
    MSE_AVERAGE = mse_average
    AVG_PRED_PROB = np.mean(predicted_prob[:, 1])

    # results

    result_fct = pd.DataFrame({"Negative_Label": [Negative_Label],
                               "Positive_Label": [Positive_Label],
                               "Label": [Label],
                               "Support_Negative": [Support_Negative],
                               "Support_Positive": [Support_Positive],
                               "TN": [TN],
                               "FP": [FP],
                               "FN": [FN],
                               "TP": [TP],
                               "Precision_Negative": [Precision_Negative],
                               "Precision_Positive": [Precision_Positive],
                               "Recall_Negative": [Recall_Negative],
                               "Recall_Positive": [Recall_Positive],
                               "Recall": [Recall],
                               "F1_Score": [F1_Score],
                               "AUC": [AUC],
                               "AUC_PR": [AUC_PR],
                               "MCC": [MCC],
                               "MSE_NEGATIVE": [MSE_NEGATIVE],
                               "MSE_POSITIVE": [MSE_POSITIVE],
                               "MSE_AVERAGE": [MSE_AVERAGE],
                               "AVG_PRED_PROB": [AVG_PRED_PROB]})

    return result_fct

    logger.info("Function ended: evaluate")


def rename_fields(dtf,
                   text_field,
                   label_field):
    logger.info("Function started: rename_fields")

    dtf["Journal"] = dtf["Source Title"]
    dtf["Journal"][dtf["Book Series Title"]=="Advances in Austrian Economics"] = dtf["Book Series Title"]

    dtf.rename(columns = {text_field:"text", label_field:"y"}, inplace=True)

    return dtf

    logger.info("Function ended: rename_fields")







def split_text(dtf,
               min_char):
    logger.info("Function started: split_text")

    dtf = dtf.loc[(dtf["text"].str.len() > min_char) ,:].copy()
    dtf['first_part'] = [str(x)[:min_char] for x in dtf.loc[:,"text"]]
    dtf['last_part'] = [str(x)[-min_char:] for x in dtf.loc[:,"text"]]

    return dtf

    logger.info("Function ended: split_text")



def detect_languages(text_series):
    #logger.info("type of text_series: " + str(text_series))
    language_series = [langdetect.detect(text_series) if text_series.strip() != "" else ""]
    return language_series



def multiprocessing_wrapper(input_data,
                            function,
                            cores):

    logger.info("CPU Cores used: " + str(cores))

    pool = mp.Pool(cores)
    results = pool.map(function, input_data)

    pool.close()
    pool.join()

    return results



def add_language_to_file(dtf,
                         language_series_beginning,
                         language_series_end):

    dtf['language_beginning'] = language_series_beginning
    dtf['language_end'] = language_series_end

    dtf.loc[dtf['language_beginning'] == dtf['language_end'], 'lang'] = dtf['language_beginning']
    dtf.loc[dtf['language_beginning'] != dtf['language_end'], 'lang'] = "unassigned"

    return dtf




def language_detection_wrapper(dtf,
                               min_char,
                               cores,
                               function):
    logger.info("Function started: language_detection_wrapper")

    #import logging
    #logger = logging.getLogger("__main__")

    dtf = split_text(dtf = dtf,
                     min_char = min_char)

    language_series_beginning = multiprocessing_wrapper(input_data = dtf['first_part'],
                                                        function = function,
                                                        cores = cores)

    language_series_end = multiprocessing_wrapper(input_data = dtf['last_part'],
                                                  function = function,
                                                  cores = cores)

    dtf = add_language_to_file(dtf = dtf,
                                 language_series_beginning = language_series_beginning,
                                 language_series_end = language_series_end)

    return dtf

    logger.info("Function ended: language_detection_wrapper")







'''
Preprocess a string.
:parameter
    :param text: string - name of column containing text
    :param lst_stopwords: list - list of stopwords to remove
    :param flg_stemm: bool - whether stemming is to be applied
    :param flg_lemm: bool - whether lemmitisation is to be applied
:return
    cleaned text
'''


def utils_preprocess_text(text,
                          flg_stemm=False,
                          flg_lemm=True,
                          remove_last_n = 50):

    lst_stopwords = nltk.corpus.stopwords.words("english")

    subtitles = ["design", "methodology", "approach", "originality", "value", "limitations", "implications", "elsevier", "purpose"]
    lst_stopwords.extend(subtitles)

    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', ' ', str(text[:-remove_last_n]).lower().strip())
    text = re.sub(r'[0-9]+', '', text)

    ## Tokenize (convert from string to list)
    lst_text = text.split()

    ## remove Stopwords
    lst_text = [word for word in lst_text if word not in lst_stopwords]

    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    ## back to string from list
    text = " ".join(lst_text)
    return text

#dtf["text_clean"] = dtf["text"].apply(lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, lst_stopwords=lst_stopwords))



def preprocessing_wrapper(dtf,
                          cores,
                          function=utils_preprocess_text):
    logger.info("Function started: preprocessing_wrapper")

    #import logging
    #logger = logging.getLogger("__main__")

    #lst_stopwords = nltk.corpus.stopwords.words("english")

    proproc_series = multiprocessing_wrapper(input_data = dtf["text"],
                                                function = function,
                                                cores = cores)

    dtf["text_clean"] = proproc_series

    return dtf

    logger.info("Function ended: preprocessing_wrapper")