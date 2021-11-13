'''
This is a reproduction of:

https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794
'''

## set up
import time

tic = time.perf_counter()

import random

random.seed(10)

import logging
import csv

##config set up
import configparser
import os
import sys


def config():
    config = configparser.ConfigParser()
    config.read(os.getcwd() + '/config.ini')

    data_path = config['PATH']['data_path']
    scripts_path = config['PATH']['scripts_path']
    project_path = config['PATH']['project_path']

    sys.path.append(project_path)
    return data_path, scripts_path, project_path


if __name__ == "__main__":
    data_path, scripts_path, project_path = config()

##parallelization
import multiprocessing as mp

'''
CLASSIFIER SPECIFIC IMPORTS
'''

## for data
import pandas as pd

pd.set_option('display.max_columns', None)

## for data
import json
import numpy as np
from scipy import stats

## for plotting
# import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

## for processing
import re
import nltk

## for language detection
import langdetect

## for bag-of-words
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing, feature_selection, metrics, linear_model, ensemble, svm

## for balancing data
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

## for explainer
from lime import lime_text
import wordcloud

## for word embedding
import gensim
import gensim.downloader as gensim_api

## for deep learning
from tensorflow.keras import models, layers, preprocessing as kprocessing

from tensorflow.keras import backend as K

## for bert language model
import transformers

############################################
logging_level = logging.INFO  # logging.DEBUG #logging.WARNING
print_charts_tables = True  # False #True
input_file_name = "WOS_lee_heterodox_und_samequality_preprocessed"
input_file_size = "all"  # 10000 #"all"
input_file_type = "csv"
output_file_name = "WOS_lee_heterodox_und_samequality_preprocessed_wip"
sample_size = "all"  # input_file_size #10000 #"all"
use_reproducible_train_test_split = True
train_set_name = "WOS_lee_heterodox_und_samequality_preprocessed_train_9"
test_set_name = "WOS_lee_heterodox_und_samequality_preprocessed_test_1"
text_field_clean = "text_clean"  # "title" #"abstract"
text_field = "text"
label_field = "y"
cores = mp.cpu_count()  # mp.cpu_count()  #2
save = False  # False #True
plot = 0  # 0 = none, 1 = some, 2 = all

save_results = True

journal_split = True
num_journals = "all"  # 3 #"all"
random_journals = False
journal_list = [i for i in range(0, 30)]  # False # [65,1]

test_size = 0.1  # suggestion: 0.1
training_set = "oversample"  # "oversample", "undersample", "heterodox", "samequality" ; suggestion: oversample

results_file_name = "results_test_tfidf_short"

# TFIDF only
tfidf = True
max_features_list = [30000]  # [1000, 5000, 10000]
p_value_limit_list = [0.7]  # [0.8, 0.9, 0.95]
ngram_range_list = [(1, 1)]  # [(1,1), (1,2), (1,3)]
tfidf_classifier_list = ["LogisticRegression"]  # ["naive_bayes", "LogisticRegression", "LogisticRegressionCV", "SVC", "RandomForestClassifier","GradientBoostingClassifier"]

# w2v only
w2v = False
use_gigaword = False  # if True the pretrained model "glove-wiki-gigaword-[embedding_vector_length]d" is used
use_embeddings = False  # if True a trained model needs to be selected below
# which_embeddings = "word2vec_numabs_79431_embedlen_300_epochs_30" #specify model to use here
embedding_folder = "embeddings"
train_new = True  # if True new embeddings are trained

num_epochs_for_embedding_list = [10]  # number of epochs to train the word embeddings ; sugegstion: 15 (embedding_set = "False")
num_epochs_for_classification_list = [10]  # number of epochs to train the the classifier ; suggetion: 10 (with 300 dim. embeddings)
embedding_vector_length_list = [300]  # suggesion: 300

window_size_list = [8]  # suggesion: 8

embedding_only = True
embedding_set = False  # "oversample", "undersample", "heterodox", "samequality", False ; suggestion: False

max_length_of_document_vector_w2v_list = [100]  # np.max([len(i.split()) for i in X_train_series]) #np.quantile([len(i.split()) for i in X_train_series], 0.7) ; suggesion: 100
classifier_loss_function_w2v_list = ['sparse_categorical_crossentropy']  # , 'mean_squared_error', 'sparse_categorical_crossentropy', "kl_divergence", 'categorical_hinge'
w2v_batch_size_list = [256]  # suggestion: 256

# BERT only
bert = False
small_model_list = [True]
bert_batch_size_list = [64]
bert_epochs_list = [6]
max_length_of_document_vector_bert_list = [350]  # np.max([len(i.split()) for i in X_train_series]) #np.quantile([len(i.split()) for i in X_train_series], 0.7) ; suggesion: 350
classifier_loss_function_bert_list = ['sparse_categorical_crossentropy']  # , 'mean_squared_error', 'sparse_categorical_crossentropy', "kl_divergence", 'categorical_hinge'
use_bert_feature_matrix = True
save_bert_feature_matrix = False

############################################

parameters = """PARAMETERS:
        input_file_name = """ + input_file_name + """
        cores = """ + str(cores) + """
        save_results = """ + str(save_results) + """
        journal_split = """ + str(journal_split) + """
        num_journals = """ + str(num_journals) + """
        random_journals = """ + str(random_journals) + """
        journal_list = """ + str(journal_list) + """
        test_size = """ + str(test_size) + """
        training_set = """ + str(training_set) + """
        use_reproducible_train_test_split = """ + str(use_reproducible_train_test_split) + """
        train_set_name = """ + str(train_set_name) + """
        test_set_name = """ + str(test_set_name) + """
        tfidf = """ + str(tfidf) + """
        w2v = """ + str(w2v) + """
        bert = """ + str(bert)

if tfidf:
    parameters_tfidf = """PARAMETERS TFIDF:
            max_features_list = """ + str(max_features_list) + """
            p_value_limit_list = """ + str(p_value_limit_list) + """
            ngram_range_list = """ + str(ngram_range_list) + """
            tfidf_classifier_list = """ + str(tfidf_classifier_list)

if w2v:
    parameters_w2v = """PARAMETERS W2V:
            use_gigaword = """ + str(use_gigaword) + """
            use_embeddings = """ + str(use_embeddings) + """
            embedding_folder = """ + str(embedding_folder) + """
            train_new = """ + str(train_new) + """
            num_epochs_for_embedding_list = """ + str(num_epochs_for_embedding_list) + """
            embedding_vector_length_list = """ + str(embedding_vector_length_list) + """
            num_epochs_for_classification_list = """ + str(num_epochs_for_classification_list) + """
            window_size_list = """ + str(window_size_list) + """
            embedding_only = """ + str(embedding_only) + """
            embedding_set = """ + str(embedding_set) + """
            max_length_of_document_vector_w2v_list = """ + str(max_length_of_document_vector_w2v_list) + """
            classifier_loss_function_w2v_list = """ + str(classifier_loss_function_w2v_list) + """
            w2v_batch_size_List = """ + str(w2v_batch_size_list)

if bert:
    parameters_bert = """PARAMETERS BERT:
            max_length_of_document_vector_bert_list = """ + str(max_length_of_document_vector_bert_list) + """
            classifier_loss_function_bert_list = """ + str(classifier_loss_function_bert_list) + """
            small_model_list = """ + str(small_model_list) + """
            bert_batch_size_list = """ + str(bert_batch_size_list) + """
            bert_epochs_list = """ + str(bert_epochs_list) + """
            use_bert_feature_matrix = """ + str(use_bert_feature_matrix) + """
            save_bert_feature_matrix = """ + str(save_bert_feature_matrix)


def monitor_process():
    ##monitor progress if run on local machine
    if not project_path[0] == "/":
        if __name__ == "__main__":
            print("--LOCAL RUN--")

            ##monitor progress
            from tqdm import tqdm
            tqdm.pandas()

    if project_path[0] == "/":
        if __name__ == "__main__":
            print("--CLUSTER RUN--")


if __name__ == "__main__":
    monitor_process()

##logs
logging.basicConfig(level=logging_level,
                    handlers=[logging.FileHandler("log.log"),
                              logging.StreamHandler()],
                    format=('%(levelname)s | '
                            '%(asctime)s | '
                            '%(filename)s | '
                            '%(funcName)s() | '
                            '%(lineno)d | \t'
                            '%(message)s'))  # , format='%(levelname)s - %(asctime)s - %(message)s - %(name)s')

logger = logging.getLogger()

logger.info(parameters)

if tfidf:
    logger.info(parameters_tfidf)

if w2v:
    logger.info(parameters_w2v)

if bert:
    logger.info(parameters_bert)

from Utils import utils_ortho_hetero as fcts

'''
LOAD DATA
'''
logger.info("LOAD DATA")
logger.info("LOAD DATA")
if __name__ == "__main__":
    dtf = fcts.load_data(data_path=data_path,
                         input_file_name=input_file_name,
                         input_file_size=input_file_size,
                         input_file_type=input_file_type,
                         sample_size="all")



else:
    all_journals = pd.DataFrame({"Journal": ["randmom"], label_field: ["random"]})
    test_journal = None

if num_journals == "all":
    last_journal = len(all_journals)
else:
    last_journal = num_journals

loop_number = 0

for index, all_test in all_journals.iterrows():
    loop_number = loop_number + 1

    if loop_number > last_journal:
        break

    logger.info("Loop Nr. = " + str(loop_number))

    logger.info("TRAIN TEST SPLIT")

    if journal_split == True:

        label = all_test[label_field]

        test_journal = all_test["Journal"]

        logger.info("Journal = " + str(test_journal))
        logger.info("Label = " + str(label))

        dtf_train = dtf.loc[dtf["Journal"] != test_journal].copy()
        dtf_test = dtf.loc[dtf["Journal"] == test_journal].copy()

        training_set_id = ''.join(test_journal.split()) + str(int(time.time() * 1000))

    else:
        if use_reproducible_train_test_split:
            dtf_train = pd.read_csv(data_path + "/" + train_set_name + ".csv")
            dtf_test = pd.read_csv(data_path + "/" + test_set_name + ".csv")
            training_set_id = "use_reproducible_train_test_split"

        else:
            dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=test_size, random_state=42)
            training_set_id = "random" + str(int(time.time() * 1000))

    # balanbce dataset
    logger.info("BALANCE TRAINING SET")

    if training_set == "oversample":
        over_sampler = RandomOverSampler(random_state=42)
        X_train, y_train = over_sampler.fit_resample(pd.DataFrame({"X": dtf_train[text_field_clean], "X_raw": dtf_train[text_field]}), pd.DataFrame({"y": dtf_train[label_field]}))
        X_train_raw = pd.DataFrame({"X": X_train["X_raw"].tolist()})
        X_train = pd.DataFrame({"X": X_train["X"].tolist()})

    elif training_set == "undersample":
        under_sampler = RandomUnderSampler(random_state=42)
        X_train, y_train = under_sampler.fit_resample(pd.DataFrame({"X": dtf_train[text_field_clean], "X_raw": dtf_train[text_field]}), pd.DataFrame({"y": dtf_train[label_field]}))
        X_train_raw = pd.DataFrame({"X": X_train["X_raw"].tolist()})
        X_train = pd.DataFrame({"X": X_train["X"].tolist()})

    else:
        X_train = pd.DataFrame({"X": dtf_train[text_field_clean], "X_raw": dtf_train[text_field]})
        y_train = pd.DataFrame({"y": dtf_train[label_field]})
        X_train_raw = pd.DataFrame({"X": X_train["X_raw"].tolist()})
        X_train = pd.DataFrame({"X": X_train["X"].tolist()})

    X_train_series = X_train.squeeze(axis=1)
    X_train_raw_series = X_train_raw.squeeze(axis=1)
    y_train_series = y_train.squeeze(axis=1)

    trainingset_id = "TRAININGSET" + str(int(time.time() * 1000))

    models_list = []

    if tfidf == True:
        models_list += ["tfidf"]

    if w2v == True:
        models_list += ["w2v"]

    if bert == True:
        models_list += ["bert"]

        if current_model == "bert":

            small_model_loop = 0

            for small_model in small_model_list:

                small_model_loop += 1

                if small_model:
                    tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
                else:
                    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

                max_length_of_document_vector_bert_loop = 0

                for max_length_of_document_vector_bert in max_length_of_document_vector_bert_list:

                    max_length_of_document_vector_bert_loop += 1
                    logger.info("Loop small_model Nr: " + str(small_model_loop))
                    logger.info("small_model : " + str(small_model))
                    logger.info("Loop max_length_of_document_vector_bert Nr: " + str(max_length_of_document_vector_bert_loop))
                    logger.info("max_length_of_document_vector_bert : " + str(max_length_of_document_vector_bert))

                    text_lst = [text[:-50] for text in X_train_raw["X"]]
                    text_lst = [' '.join(text.split()[:max_length_of_document_vector_bert]) for text in text_lst]

                    subtitles = ["design", "methodology", "approach", "originality", "value", "limitations", "implications"]

                    text_lst = [word for word in text_lst if word not in subtitles]

                    # text_lst = [text for text in text_lst if text]

                    corpus = text_lst

                    ## Fearute engineering train set
                    logger.info("Fearute engineering train set")

                    ## add special tokens
                    logger.info("add special tokens")

                    maxqnans = np.int((max_length_of_document_vector_bert - 20) / 2)
                    corpus_tokenized = ["[CLS] " +
                                        " ".join(tokenizer.tokenize(re.sub(r'[^\w\s]+|\n', '',str(txt).lower().strip()))[:maxqnans]) +
                                        " [SEP] " for txt in corpus]

                    ## generate masks
                    logger.info("generate masks")
                    masks = [[1] * len(txt.split(" ")) + [0] * (max_length_of_document_vector_bert - len(txt.split(" "))) for txt in corpus_tokenized]

                    ## padding
                    logger.info("padding")
                    txt2seq = [txt + " [PAD]" * (max_length_of_document_vector_bert - len(txt.split(" "))) if len(txt.split(" ")) != max_length_of_document_vector_bert else txt for txt in corpus_tokenized]

                    ## generate idx
                    logger.info("generate idx")


                    idx = [tokenizer.encode(seq.split(" "), is_split_into_words=True) for seq in txt2seq]
                    minlen = min([len(i) for i in idx])
                    idx = [i[:max_length_of_document_vector_bert] for i in idx]

                    np.savetxt(data_path + "/" + str(train_set_name) + "_" + str(max_length_of_document_vector_bert) + "_bert_feature_matrix.csv",
                               idx,
                               delimiter=",",
                               fmt='% s')