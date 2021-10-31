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
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing, \
    feature_selection, metrics

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

'''
from tensorflow.keras import backend as K

## for bert language model
import transformers
'''

############################################
logging_level = logging.INFO  # logging.DEBUG #logging.WARNING
print_charts_tables = True  # False #True
input_file_name = "WOS_lee_heterodox_und_samequality_preprocessed"
input_file_size = "all"  # 10000 #"all"
input_file_type = "csv"
output_file_name = "WOS_lee_heterodox_und_samequality_preprocessed_wip"
sample_size = "all"  # input_file_size #10000 #"all"
text_field_clean = "text_clean"  # "title" #"abstract"
text_field = "text"
label_field = "y"
cores = mp.cpu_count()  # mp.cpu_count()  #2
save = False  # False #True
plot = 0  # 0 = none, 1 = some, 2 = all
use_gigaword = True  # if True the pretrained model "glove-wiki-gigaword-[embedding_vector_length]d" is used
use_embeddings = False  # if True a trained model needs to be selected below
# which_embeddings = "word2vec_numabs_79431_embedlen_300_epochs_30" #specify model to use here
embedding_folder = "embeddings"
train_new = False  # if True new embeddings are trained
num_epochs_for_embedding_list = [
    15]  # number of epochs to train the word embeddings ; sugegstion: 10(-15) (undersampling training)
num_epochs_for_classification_list = [
    10]  # number of epochs to train the the classifier ; suggetion: 10 (with 300 dim. embeddings)
embedding_vector_length_list = [300]  # suggesion: 300
window_size_list = [8]  # suggesion: 8
max_length_of_document_vector = 100  # np.max([len(i.split()) for i in X_train_series]) #np.quantile([len(i.split()) for i in X_train_series], 0.7) ; suggesion: 8
embedding_only = False
save_results = True
results_file_name = "tfidf_classifier_performance_per_journal_all.csv"
test_size = 0.1  # suggestion: 0.1
training_set = "oversample"  # "oversample", "undersample", "heterodox", "samequality" ; suggestion: oversample
embedding_set = False  # "oversample", "undersample", "heterodox", "samequality", False ; suggestion: False

sentiment = False
max_features = 1000
############################################

parameters = """PARAMETERS:
input_file_name = """ + input_file_name + """
cores = """ + str(cores) + """
use_gigaword = """ + str(use_gigaword) + """
use_embeddings = """ + str(use_embeddings) + """
embedding_folder = """ + str(embedding_folder) + """
train_new = """ + str(train_new) + """
num_epochs_for_embedding_list = """ + str(num_epochs_for_embedding_list) + """
num_epochs_for_classification_list = """ + str(num_epochs_for_classification_list) + """
embedding_vector_length_list = """ + str(embedding_vector_length_list) + """
window_size_list = """ + str(window_size_list) + """
max_length_of_document_vector = """ + str(max_length_of_document_vector) + """
embedding_only = """ + str(embedding_only) + """
save_results = """ + str(save_results) + """
results_file_name = """ + str(results_file_name) + """
test_size = """ + str(test_size) + """
training_set = """ + str(training_set) + """
embedding_set = """ + str(embedding_set)


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

'''
results_file = pd.DataFrame({"training_length":[],
                             "use_gigaword":[],
                             "num_epochs_for_embedding":[],
                             "num_epochs_for_classification":[],
                             "embedding_vector_length":[],
                             "window_size":[],
                             "max_length_of_document_vector":[],
                             "training_set":[],
                             "embedding_set":[],
                             "test_size": [],
                             "Negative_Label":[],
                             "Positive_Label":[],
                             "Support_Negative":[],
                             "Support_Positive":[],
                             "TN":[],
                             "FP":[],
                             "FN":[],
                             "TP":[],
                             "Precision_Negative":[],
                             "Precision_Positive":[],
                             "Recall_Negative":[],
                             "Recall_Positive":[],
                             "AUC":[],
                             "AUC-PR":[],
                             "MCC":[]})
'''

if use_gigaword + use_embeddings + train_new != 1:
    sys.exit(
        "invalid parameter setting: set only of use_gigaword, use_embeddings and which_embeddings to 'True' and the other two to 'False'")

from Utils import utils_ortho_hetero as fcts

'''
LOAD DATA
'''
logger.info("LOAD DATA")
if __name__ == "__main__":
    dtf = fcts.load_data(data_path=data_path,
                         input_file_name=input_file_name,
                         input_file_size=input_file_size,
                         input_file_type=input_file_type,
                         sample_size="all")

if plot == 1 or plot == 2:
    fig, ax = plt.subplots()
    fig.suptitle("Label Distribution in Original Data", fontsize=12)
    dtf[label_field].reset_index().groupby(label_field).count().sort_values(by="index").plot(kind="barh", legend=False,
                                                                                             ax=ax).grid(axis='x')
    plt.show()

'''
TRAIN TEST SPLIT
'''

gigaword_loaded = False  # dont change this
all_journals = dtf[["Journal", label_field]].drop_duplicates().copy()

loop_number = 0

for index, all_test in all_journals.iterrows():

    loop_number = loop_number + 1

    logger.info("Loop Nr. = " + str(loop_number))

    logger.info("TRAIN TEST SPLIT")

    label = all_test[label_field]

    test_journal = all_test["Journal"]

    logger.info("Journal = " + str(test_journal))

    dtf_train = dtf.loc[dtf["Journal"] != test_journal].copy()
    dtf_test = dtf.loc[dtf["Journal"] == test_journal].copy()

    # balanbce dataset
    logger.info("BALANCE TRAINING SET")

    if training_set == "oversample":
        over_sampler = RandomOverSampler(random_state=42)
        X_train, y_train = over_sampler.fit_resample(pd.DataFrame({"X": dtf_train[text_field_clean]}),
                                                     pd.DataFrame({"y": dtf_train[label_field]}))

    elif training_set == "undersample":
        under_sampler = RandomUnderSampler(random_state=42)
        X_train, y_train = under_sampler.fit_resample(pd.DataFrame({"X": dtf_train[text_field_clean]}),
                                                      pd.DataFrame({"y": dtf_train[label_field]}))





    logger.info("TFIDF VECTORIZER")

    vectorizer = feature_extraction.text.TfidfVectorizer(max_features=max_features, ngram_range=(1,2))

    corpus = X_train["X"]

    vectorizer.fit(corpus)
    X_train = vectorizer.transform(corpus)
    dic_vocabulary = vectorizer.vocabulary_

    if plot == 1 or plot == 2:
        sns.heatmap(X_train.todense()[:,np.random.randint(0,X_train.shape[1],100)]==0, vmin=0, vmax=1, cbar=False).set_title('Sparse Matrix Sample')




    #FEATURE SELECTION
    logger.info("FEATURE SELECTION")
    y = y_train
    X_names = vectorizer.get_feature_names()
    p_value_limit = 0.95

    dtf_features = pd.DataFrame()
    for cat in np.unique(y):
        chi2, p = feature_selection.chi2(X_train, y==cat)
        dtf_features = dtf_features.append(pd.DataFrame(
                       {"feature":X_names, "score":1-p, "y":cat}))
        dtf_features = dtf_features.sort_values([label_field,"score"],
                        ascending=[True,False])
        dtf_features = dtf_features[dtf_features["score"]>p_value_limit]

    X_names = dtf_features["feature"].unique().tolist()


    '''
    for cat in np.unique(y):
       print("# {}:".format(cat))
       print("  . selected features:",
             len(dtf_features[dtf_features["y"]==cat]))
       print("  . top features:", ",".join(
    dtf_features[dtf_features["y"]==cat]["feature"].values[:10]))
       print(" ")
    '''


    #shorter
    logger.info("SHORTENING VOCABULARY")
    vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)

    vectorizer.fit(corpus)
    X_train = vectorizer.transform(corpus)
    dic_vocabulary = vectorizer.vocabulary_


    #classify
    logger.info("SET UP CLASSIFIER")

    classifier = naive_bayes.MultinomialNB()

    ## pipeline
    model = pipeline.Pipeline([("vectorizer", vectorizer),
                               ("classifier", classifier)])

    ## train classifier
    logger.info("TRAIN CLASSIFIER")

    y_train = y_train.values.ravel()

    model["classifier"].fit(X_train, y_train)


    ## test
    logger.info("TEST CLASSIFIER")

    X_test = dtf_test[text_field_clean].values

    ''' PLAUSIBIBILITY CHECK
    X_test = X_train_vector.values.ravel()
    '''

    predicted = model.predict(X_test)
    predicted_prob = model.predict_proba(X_test)

    y_test = dtf_test[label_field].values

    ''' PLAUSIBIBILITY CHECK
    y_test = y_train_vector.values.ravel()
    '''

    classes = np.unique(y_test)
    y_test_array = pd.get_dummies(y_test, drop_first=False).values








    correct_classifications = sum([pred == label for pred in predicted])
    incorrect_classifications = sum([pred != label for pred in predicted])

    result = pd.DataFrame({"name": [test_journal],
                           "label": [label],
                           "correct_classifications": [correct_classifications],
                           "incorrect_classifications": [incorrect_classifications]})

    if save_results:
        results_path = data_path + "/" + results_file_name
        results = pd.read_csv(results_path)
        results = pd.concat([results, result])
        results.to_csv(results_path, index=False)

toc = time.perf_counter()
logger.info(f"whole script for {len(dtf)} in {toc - tic} seconds")
print("the end")