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


PREREQUISITS:
- GET DATA FROM THE ABOVE SOURCES OR FROM THE WEB OF SCIENCE
- IF YOU DONT USE THE PRE-PROCESSED DATA FROM THE LINK ABOVE, APPLY THE PROPROCESSING.PY SCRIPT FROM THE ABOVE GITHUP REPO TO YOUR DATA

THIS SCRIPT IS USED TO SELECT, TRAIN, EVALUATE AND APPLY THE MODELS DESCRIBED IN THE MASTERS THESIS "THE DIALECTS OF ECONOSPEAK" BY DAMIAN DURRER (2021)
- USE SECTION 0.1 TO REPRODUCE RESULTS FRO THESIS
- USE SECTION 0.2 TO CHOOSE YOUR OWN PARAMETER SETTINGS
'''




'''
PACKAGES & SET UP
'''
#######################################
## set up
import time
tic = time.perf_counter()

import random
random.seed(10)

import logging

import pickle

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

## for data
import pandas as pd

pd.set_option('display.max_columns', None)

## for data
import numpy as np

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns

## for processing
import re

## for bag-of-words
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing, feature_selection, metrics, linear_model, ensemble, svm

## for balancing data
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

## for explainer
from lime import lime_text

## for word embedding
import gensim
import gensim.downloader as gensim_api

## for deep learning
from tensorflow.keras import models, layers, preprocessing as kprocessing

## for bert language model
import transformers
#######################################


'''
----------------------------------------------------------
DEFAULT PARAMETERS
----------------------------------------------------------
'''
#######################################
logging_level = logging.INFO
cores = mp.cpu_count()
plot = 0
input_file_name = ""
text_field_clean = "text_clean"
text_field = "text"
label_field = "y"
journal_field = "Journal"
save_results = False
results_file_name = False
save_weights = False
current_model = False
min_df_list = []
p_value_limit_list = []
ngram_range_list = []
tfidf_classifier_list = []
embedding_vector_length_list = []
train_new = False
embedding_only = False
num_epochs_for_embedding_list = []
window_size_list = []
embedding_set = False
use_gigaword = False
use_embeddings = False
embedding_folder = "embeddings"
which_embeddings = False
num_epochs_for_classification_list = []
max_length_of_document_vector_w2v_list = []
classifier_loss_function_w2v_list = []
w2v_batch_size_list = []
small_model_list = []
bert_batch_size_list = []
bert_epochs_list = []
max_length_of_document_vector_bert_list = []
classifier_loss_function_bert_list = []
use_bert_feature_matrix = False
save_bert_feature_matrix = False
use_model = False
model_file_name = False
save_model = False
train_on_all = False
training_set = False
journal_split = False
use_reproducible_train_test_split = False
train_set_name = ""
test_set_name = ""
test_size = 0.1
num_journals = "all"
journal_list = []
#######################################





'''
#########################################
SECTION 0.1
#########################################

----------------------------------------------------------
REPRODUCE RESULTS
----------------------------------------------------------

Load parameters to reproduce results
----------------------------------------------------------
'''

## reproduce results
load_parameter_settings = True

if load_parameter_settings:

    ####
    # REPRODUCE MODEL SELECTION
    ###

    # from Utils.REPRODUCE_1Model_Selection_TFIDF import *
    # from Utils.REPRODUCE_1Model_Selection_W2V import *
    # from Utils.REPRODUCE_1Model_Selection_BERT import *



    ####
    # REPRODUCE JOURNAL BY JOURNAL EVALUATION
    ###

    # from Utils.REPRODUCE_4Journals_TFIDF import *
    # from Utils.REPRODUCE_4Journals_W2V import *
    # from Utils.REPRODUCE_4Journals_BERT import *



    ####
    # REPRODUCE FINAL MODEL TRAINING ON ALL DATA AND WEIGHTS LISTS
    ###

    # from Utils.REPRODUCE_3All_Data_TFIDF import *
    # from Utils.REPRODUCE_3All_Data_W2V import *
    # from Utils.REPRODUCE_3All_Data_BERT import *


    ####
    # REPRODUCE APPLICATION OF MODELS TO TOP 5 JOURNALS
    ###

    # from Utils.REPRODUCE_5Top5_TFIDF import *
    # from Utils.REPRODUCE_5Top5_W2V import *
    # from Utils.REPRODUCE_5Top5_BERT import *
    print()


'''
#########################################
SECTION 0.2
#########################################

SET YOUR OWN PARAMETERS
----------------------------------------------------------
'''

## INPUT REQUIRED IF load_parameter_settings = False!!!
if not load_parameter_settings:

    #############################################################
    # settings
    #############################################################

    logging_level = logging.INFO
    # Controlling the outputs printed in the console
    #### Recommended: logging.INFO
    #### Alternatives: logging.DEBUG ; logging.WARNING

    cores = mp.cpu_count()
    # Controlling the number of cores used in computations
    #### Recommended: mp.cpu_count() // all cores
    #### Alternatives: 1,2, ... // specify number of cores

    plot = 0
    # Controlling analytics plots to be created
    #### Recommended: 0 // no plots
    #### Alternatives: 1 // some plots : 2 // all plots





    #############################################################
    # input file
    #############################################################

    input_file_name = "WOS_lee_heterodox_und_samequality_new_preprocessed_2_1000"
    ## File needs to be located under the DATA_PATH and include the below specified columns (w/out ".csv"):
    #### Recommended: Full Dataset: "WOS_lee_heterodox_und_samequality_new_preprocessed_2" // for model training and performance evaluation
    #### Alternative:  Top 5 Dataset: "WOS_top5_new_preprocessed_2" // application
    #### Alternative:  Short dataset to test code: "WOS_lee_heterodox_und_samequality_new_preprocessed_2_1000" // development

    text_field_clean = "text_clean"  # "title" #"abstract"
    ## column name of the pre-processed abstracts in the input file // use SCRIPTS_PATH/preprocessing.py // for TFIDF and WORD2VEC
    #### Recommended: 'text_clean'

    text_field = "text"
    ## column name for the raw abstracts in the input file // for BERT
    #### Recommended: 'text'

    label_field = "y"
    ## column name for the labels {"0samequality", "1heterodox"} in the input file // for training, not required if pre-trained models are applied
    #### Recommended: 'y'
    #### Alternative: 'labels'

    journal_field = "Journal"
    ## column name for the journals in the input file
    #### Recommended: 'Journal'




    #############################################################
    # save results, models, weights and training-samlpes
    #############################################################
    save_results = False
    ## False if results should not be saved, True if they should be saved // OVERWRITES PREVIOUS FILES WITH THE SAME NAME

    results_file_name = False
    ## Save the results to a file  in DATA_PATH/results // OVERWRITES PREVIOUS FILES WITH THE SAME NAME
    #### Recommended: False // automatic name is set based on parameter choices
    #### Alternative: e.g. "5TOP5_BERT" // Chose file name according to the settings ("TRAINING", "JOURNALS", "TOP5") and the models used ("TFIDF","W2V", "BERT")

    save_weights = False
    ## save TFIDF weights to DATA_PATH/weights // name is automatically assigned accorting to the model name and input data name





    #############################################################
    # Set model hyperparameters and settings
    #############################################################

    current_model = "tfidf"
    ## choose which type of model to use: "tfidf", "w2v" or "bert"
    #### Recommended: "tfidf" // tfidf weighting of features with subsequent classification (classifier can be selected below)
    #### Alternative: "w2v" // word embeddings with subsequent LSTM classification; "bert" // bert transformer model

    #### FOR ALL THE BELOW:
    #### Provide parameters as list. If multiple parameters are provided: grid search over all combination is performed

    if current_model == "tfidf":
        min_df_list = [5]
        ## choose the minimum document frequency for the terms (only terms with document frequency > min_df are included in the feature matrix
        #### Recommended: [5]
        #### Alternative: [3, 10]

        p_value_limit_list = [0.0]  # [0.8, 0.9, 0.95]
        ## choose the minimum p-value with which a term has to be correlated to the label (only terms with a p-value of larger than  p_value_limit are included)
        #### Recommended: [0.0] // no p-value limit applied
        #### Alternative: [0.9, 0.95] //

        ngram_range_list = [(1, 1)]
        ## choose range of n-grams to include as features
        #### Recommended: [1,1] // only words (1-grams)
        #### Alternative: [1,3] // 1-, 2-, and 3-grams are included; any other range is possible

        tfidf_classifier_list = ["LogisticRegression"]  # ["naive_bayes", "LogisticRegression", "RandomForestClassifier","GradientBoostingClassifier", "SVC"]
        ## choose a classifier
        #### Recommended: ["LogisticRegression"] // only words (1-grams)
        #### Alternative: ["naive_bayes", "RandomForestClassifier","GradientBoostingClassifier", "SVC"]


    if current_model == "w2v":

        embedding_vector_length_list = [300]
        ## choose length of the embedding vectors (to train or load)
        #### Recommendation: [300] // 300-dimensional embeddings
        #### Alternative: [50, 150]

        train_new = False
        ##choose whether you want to train new embeddings

        embedding_only = False
        ## choose whether to only train the embeddings without the classification step (used to only store embeddings to use for classification later)
        #### Recommendation : False // perform classification after training the embeddings
        #### Alternative : True // only train embedding without classification (used to only store embeddings



        if train_new:
            num_epochs_for_embedding_list = [5]
            ## choose number of epochs to train the word embeddings
            #### Recommendation: [5]
            #### Alternative: [10,15]

            window_size_list = [12]
            ## choose window-size to use in for training the embeddings
            #### Recommendation: [12]
            #### Alternative: [4, 8]

            embedding_set = False
            ## choose whether or not to balance the training set before training the embeddings (independent of whether it is balanced for the classification)
            #### Recommendation: False // no balancing of the training set for the embeddings
            #### Alternative: "oversample" // apply random oversampling of the minority class;  "undersamling" // apply random undersampling of the majority class

        if not train_new:
            use_gigaword = False
            ## chose whether to use the pretrained embeddings from "glove-wiki-gigaword-[embedding_vector_length]d" --> IF TRUE, NO EMBEDDINGS WILL BE TRAINED ON YOUR DATA
            #### Recommended: False // dont use pretrained embeddings
            #### Alternative: True

            if not use_gigaword:
                use_embeddings = False
                ## choose your own pre-trained embeddings if True
                ####Recommended: False // train your own embeddings
                ####Alternative: True // use pretrained embeddigns --> specify source in "which_emneddings"

                if use_embeddings:
                    embedding_folder = "embeddings"
                    which_embeddings = False
                    ##specify path where embeddings are stored (under DATA_PATH/[embedding_folder]/[which_embeddings]

        if not embedding_only:
            num_epochs_for_classification_list = [15]
            ## choose number of epochs to train the the classifier
            #### Recommendation: [15]
            #### Alternative: [5, 10]

            max_length_of_document_vector_w2v_list = [100]
            ## choose the maximum length of the document vector, i.e. the max. number of words of any document in the corpus to include (truncated after this number is reached)
            #### Recommendation: [100]
            #### Alternative: [80, 150]

            classifier_loss_function_w2v_list = ['sparse_categorical_crossentropy']
            ## choose the loss function to use in the Bi LSTM for classification
            #### Recommendation: ['sparse_categorical_crossentropy']
            #### Alternative: ['mean_squared_error', 'sparse_categorical_crossentropy', "kl_divergence", 'categorical_hinge']

            w2v_batch_size_list = [256]  # suggestion: 256
            ## choose the batch-size for training the BiLSTM
            #### Recommendation: [256]
            #### Alternative: [64, 128, 512]


    if current_model == "bert":
        small_model_list = [True]
        ## choose whether to use Distilbert or Bert
        #### Recommendation: [True] // use distilbert_uncased (smaller, faster)
        #### Alternative:  [False] // use full bert_uncased (larger, slower)


        bert_batch_size_list = [64]
        ## choose the batch size to train the transformer
        #### Recommendation: [64]
        #### Alternative:  [128, 254] // potentially memory issue on GPUs (depending on hardware)

        bert_epochs_list = [12]
        ## choose for how many epochs to train the transformer model
        #### Recommendation: [12]
        #### Alternative:  [3, 6, 18, 24] // time increases/decreases more or less linearly in the number of epochs

        max_length_of_document_vector_bert_list = [200]
        ## choose maximum number of tokens to from each document to include (choose larger than the number of words because words are split into multiple tokens).
        #### Recommendation: [200] // truncating tokenized documents after 200 tokens
        #### Alternative:  [150, 300]

        classifier_loss_function_bert_list = ['sparse_categorical_crossentropy']
        ## choose the loss function to use in the Bi LSTM for classification
        #### Recommendation: ['sparse_categorical_crossentropy']
        #### Alternative: ['mean_squared_error', 'sparse_categorical_crossentropy', "kl_divergence", 'categorical_hinge']


        use_bert_feature_matrix = False
        ## choose whether to load a pre-existing bert-feature matrix (saved under "data_path/[input_file_name]_[max_length_of_document_vector_bert]_bert_feature_matrix.csv")
        #### Recommendation: False
        #### Alternative: True (load from "data_path/[input_file_name]_[max_length_of_document_vector_bert]_bert_feature_matrix.csv")

        save_bert_feature_matrix = False
        ## choose whether to save thebert-feature matrix (save to "data_path/[input_file_name]_[max_length_of_document_vector_bert]_bert_feature_matrix.csv")
        ## --> PREVIOUS VERSIONS ARE OVERWRITTEN!
        #### Recommendation: False
        #### Alternative: True (save to "data_path/[input_file_name]_[max_length_of_document_vector_bert]_bert_feature_matrix.csv")






    #############################################################
    # apply existing model or train new model?
    #############################################################

    use_model = True
    ## False if results should not be saved, True if they should be saved // OVERWRITES PREVIOUS FILES WITH THE SAME NAME
    #### Recommended: True (use existing model on new data) ; False (train new model)
    #### Note: if model is loaded: settings from above




    #############################################################
    # location of model file name
    #############################################################

    model_file_name = False
    ## Load the model from - or save the model to - a file or folder in DATA_PATH/models// OVERWRITES PREVIOUS FILES WITH THE SAME NAME
    #### Recommended: False // automatic name is set based on parameter choices
    #### Alternative: e.g. "5TOP5_BERT" // Chose file name from which you want to load the model from or save the model to



    #############################################################
    # settings train a new model
    #############################################################

    if not use_model:
        save_model = False
        ## False if model should not be saved for later use, True if it should be saved // OVERWRITES PREVIOUS FILES WITH THE SAME NAME

        train_on_all = False
        ## Choose whether to train (and test) on the whole dataset --> RESULTS WILL OVERSTATE PERFORMANCE
        #### Recommended: False // only use a part of the data for training and evaluate the results on a test set
        #### Alternative: True // train a final model on all data

        training_set = "oversample"
        ## Choose whether or not to balance the training set
        #### Recommendation: "oversample" // apply random oversampling to the the minority class to balance the training set
        #### "undersample" // apply random undersampling of the majority class
        #### False // no balancing

        if not train_on_all:
            journal_split = False
            ## select whether to apply cross validation by holding out the articles from one journal at a time
            #### Recommended: False // no cross validation - test- and train-set will include articles from all journals
            #### Alternative: True // fit n models (where n is the number of journals) and evaluate each model on the hold-out journal

            if not journal_split:
                use_reproducible_train_test_split = False
                ## Choose whether a reproducible train-test split to train models or create a new split
                #### Recommended: True // for reproducible results based on previous split ; False // for new, random train-test split should

                if use_reproducible_train_test_split:
                    train_set_name = "WOS_lee_heterodox_und_samequality_preprocessed_train_9"
                    test_set_name = "WOS_lee_heterodox_und_samequality_preprocessed_test_1"
                    ## Choose file names of the train/test files (w/out ".csv")

                if not use_reproducible_train_test_split:
                    test_size = 0.1
                    ## select the fraction of data to hold out as test set
                    #### Recommended: 0.1
                    #### Alternative: any number between in (0,1]

            if journal_split:
                num_journals = "all"  # 3 #"all"
                ## select the number of journals to use as hold-out journals (i.e. the number of loops to perform)
                #### Recommended: "all" // use each journals at hold-out set once
                #### Alternative: 1,2,3,4,5,... // choose any number k of journals

                journal_list = []  # False # [65,1]
                ## select specific list of journals to use as hold out set (one after another), by name or number
                #### Recommended: False // use each journal once
                #### Alternative: ["Cambridge Journal of Economics", "Journal of Economic Issues"] // choose a list of journals to use as hold out set (one by one)
                #### Alternative [i for i in range(0, 5)] // choose a list of journals to use as hold out set (one by one)
                #### Alternative [1,5,8,66] // choose a list of journals to use as hold out set (one by one)
#######################################

#set the model dummy parameters
#######################################
tfidf = w2v = bert = False
if current_model == "tfidf":
    tfidf = True
if current_model == "w2v":
    w2v = True
if current_model == "bert":
    bert = True
#######################################

#create string with all parameters for logs
#######################################

## save parameter strings to print to log
parameters = """PARAMETERS:
input_file_name = """ + input_file_name + """
cores = """ + str(cores) + """
save_results = """ + str(save_results) + """
journal_split = """ + str(journal_split) + """
num_journals = """ + str(num_journals) + """
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
    min_df_list = """ + str(min_df_list) + """
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
#######################################


#monitor progress if script is run locally
#######################################
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
#######################################

##logger settings
#######################################
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
#######################################

#set results file names and model file names
#######################################
if results_file_name == False:
    if tfidf:
        results_file_name = input_file_name + "_tfidf_results"
    if w2v:
        results_file_name = input_file_name + "_w2v_results"
    if bert:
        results_file_name = input_file_name + "_bert_results"
#######################################
#######################################
if model_file_name == False:

    if tfidf:
        model_file_name = input_file_name + "_tfidf_model"
    if w2v:
        model_file_name = input_file_name + "_w2v_model"
    if bert:
        model_file_name = input_file_name + "_bert_model"
#######################################

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

'''
#######################################
SECTION 1
#######################################

Use trained model on a new dataset
-------------------------------------------------------------------
'''

if use_model:

    ## load data
    logger.info("LOAD DATA")

    dtf = pd.read_csv(data_path + "/" + input_file_name + ".csv")

    dtf["index"] = np.arange(dtf.shape[0]) #create index column

    X_test = dtf[text_field_clean].values #define full data as test set (since no training/testing split)

    ## map 0 to the orthodox category, 1 the heterodox"
    dic_y_mapping = {0: "0orthodox",
                     1: "1heterodox"}
    inverse_dic = {v: k for k, v in dic_y_mapping.items()}


    ## apply TFIDF model
    if tfidf:
        logger.info("APPLY EXISTING TFIDF MODEL TO DATA")

        ##load model file
        model_file_path = (data_path + "/models/" + model_file_name + ".pkl")

        with open(model_file_path, 'rb') as file:
            model_loaded = pickle.load(file)

        ## apply model
        logger.info("APPLY MODEL")
        predicted = model_loaded.predict(X_test)
        predicted_prob = model_loaded.predict_proba(X_test)
        predicted_bin = np.array([np.argmax(pred) for pred in predicted_prob])

        ## save model weights
        if save_weights:
            logger.info("SAVE WEIGHTS")

            ##load vocabulary with weights from existing model
            weights_loaded_path = data_path + "/weights/" + model_file_name + "_tfidf_weights.csv"

            weights_loaded_dtf = pd.read_csv(weights_loaded_path)
            weights_loaded_dict = {k: v for k, v in zip(weights_loaded_dtf["words"], weights_loaded_dtf["weights"])}

            ##count how often the terms occur on the papers classified as orthodox and heterodox separately
            corpus = dtf[text_field_clean]
            dtf_weights = pd.DataFrame()
            for prediction in range(2):
                idx = [idx for idx, i in enumerate(predicted_bin) if i == prediction]
                corpus_loop = corpus[idx]
                ## create list of n-grams
                lst_corpus = []
                for string in corpus_loop:
                    lst_words = string.split()
                    lst_grams = [" ".join(lst_words[i:i + 1]) for i in range(0, len(lst_words), 1)]
                    lst_corpus.append(lst_grams)

                dtf_corpus = pd.DataFrame({"words": [i for l in lst_corpus for i in l]})
                dtf_corpus = dtf_corpus["words"].value_counts()


                def catch(dict, w):
                    try:
                        return dict[w]
                    except:
                        return None


                weights_list = [catch(weights_loaded_dict, w) for w in dtf_corpus.index.tolist()]
                dtf_weights = dtf_weights.append(pd.DataFrame({"words": dtf_corpus.index.tolist(),
                                                               "counts": dtf_corpus.values.tolist(),
                                                               "weights": weights_list,
                                                               "prediction": [prediction for i in weights_list]})).copy()

            ##save the weights, counts and weighted counts for interpretation to file
            dtf_weights["weighted_counts"] = dtf_weights["counts"] * dtf_weights["weights"]
            dtf_weights.to_csv(data_path + "/weights/input_" + input_file_name + "_model_" + model_file_name + "_tfidf_weights.csv", index=False)



    ## apply TFIDF model
    if w2v:
        logger.info("APPLY EXISTING W2V MODEL TO DATA")

        ##load model file
        model_file_path = (data_path + "/models/" + model_file_name)

        model_loaded = models.load_model(model_file_path)

        ## create list of n-grams
        logger.info("CREATE LIST OF N-GRAMS")
        corpus = dtf[text_field_clean]

        lst_corpus = []
        for string in corpus:
            lst_words = string.split()
            lst_grams = [" ".join(lst_words[i:i + 1]) for i in range(0, len(lst_words), 1)]
            lst_corpus.append(lst_grams)


        ## loading tokenizer
        logger.info("LOADING TOKENIZER")
        tokenizer_file_path = (data_path + "/models/" + model_file_name + "_tokenizer.pkl")
        with open(tokenizer_file_path, 'rb') as file:
            tokenizer = pickle.load(file)

        #### NEXT LINES FROM https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794
        ## detect bigrams and trigrams
        bigrams_detector = gensim.models.phrases.Phrases(lst_corpus, delimiter=' ', min_count=5, threshold=10)
        bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
        trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[lst_corpus], delimiter=" ", min_count=2, threshold=10)
        trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)

        ## detect common bigrams and trigrams using the fitted detectors
        lst_corpus = list(bigrams_detector[lst_corpus])
        lst_corpus = list(trigrams_detector[lst_corpus])

        ## text to sequence with the fitted tokenizer
        lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)

        ## padding sequence
        max_length_of_document_vector_w2v = max_length_of_document_vector_w2v_list[0]
        X_test = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=max_length_of_document_vector_w2v, padding="post", truncating="post")

        ## apply w2v model
        logger.info("APPLY MODEL")
        predicted_prob = model_loaded.predict(X_test)
        predicted = [dic_y_mapping[np.argmax(pred)] for pred in predicted_prob]
        predicted_bin = np.array([np.argmax(pred) for pred in predicted_prob])


        """
        ## LOAD ATTENTION LAYER: BASED ON https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794
        
        layer = [layer for layer in model_loaded.layers if "attention" in layer.name][0]
        func = K.function([model_loaded.input], [layer.output])

        weights = np.zeros((len(X_test), max_length_of_document_vector_w2v))
        for i in range(0, len(X_test), 2500):
            weight = func(X_test[i:i + 2500])[0]
            weight = np.mean(weight, axis=2)
            weights[i:i + 2500] = weight

        # weights = np.mean(weights, axis=2).flatten()

        ### 3. rescale weights, remove null vector, map word-weight
        weights = [preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(np.array(i).reshape(-1, 1)).reshape(-1) for i in weights]

        inv_word_dic = {v: k for k, v in tokenizer.word_index.items()}

        dtf_weights_long = pd.DataFrame()
        for prediction in range(2):
            idx = [idx for idx, i in enumerate(predicted_bin) if i == prediction]
            weights_loop = [weights[i] for i in idx]
            weights_loop = [i for s in weights for i in s]
            words = [X_test[i] for i in idx]
            words = [i for s in words for i in s]
            weights_loop = [weights_loop[n] for n, idx in enumerate(words) if idx > 1]
            words = [inv_word_dic[i] for i in words if i > 1]
            dtf_weights_long = dtf_weights_long.append(pd.DataFrame({"words": words, "weights": weights_loop, "prediction": [prediction for i in words]})).copy()
        dtf_weights = dtf_weights_long.groupby(["words", "prediction"]).mean().copy()
        dtf_weights["count"] = dtf_weights_long.groupby(["words", "prediction"]).count()["weights"].to_list()

        dtf_weights.to_csv(data_path + "/weights/input_" + input_file_name + "_model_" + model_file_name + "_w2v_weights.csv")
        """



    ## apply TFIDF model
    if bert:
        logger.info("APPLY EXISTING BERT MODEL TO DATA")

        ##load model file
        model_file_path = (data_path + "/models/" + model_file_name)

        model_loaded = models.load_model(model_file_path)

        small_model = small_model_list[0]

        ## load the tokenizer
        logger.info("LOAD TOKENIZER")
        if small_model:
            ## DistilBert
            try:
                tokenizer = transformers.AutoTokenizer.from_pretrained(data_path + '/distilbert-base-uncased/', do_lower_case=True)
            except:
                tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        else:
            ## Bert
            try:
                tokenizer = transformers.AutoTokenizer.from_pretrained(data_path + '/bert-base-uncased/', do_lower_case=True)
            except:
                tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


        #Save data as test-set (because no train-test split)
        logger.info("Feature engineer Test set")

        #### NEXT LINES FROM https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794

        X_test_ids = dtf["index"].tolist()
        max_length_of_document_vector_bert = max_length_of_document_vector_bert_list[0]
        maxqnans = np.int((max_length_of_document_vector_bert - 5))

        text_lst = [text[:-50] for text in dtf[text_field]]
        text_lst = [' '.join(text.split()[:maxqnans]) for text in text_lst]

        corpus = text_lst

        ## add special tokens
        logger.info("add special tokens test")
        corpus_tokenized = ["[CLS] " +
                            " ".join(tokenizer.tokenize(re.sub(r'[^\w\s]+|\n', '', str(txt).lower().strip()))[:maxqnans]) +
                            " [SEP] " for txt in corpus]

        ## generate masks
        logger.info("generate masks test")
        masks = [[1] * len(txt.split(" ")) + [0] * (max_length_of_document_vector_bert - len(txt.split(" "))) for txt in corpus_tokenized]

        ## padding
        logger.info("padding test")
        txt2seq = [txt + " [PAD]" * (max_length_of_document_vector_bert - len(txt.split(" "))) if len(txt.split(" ")) != max_length_of_document_vector_bert else txt for txt in corpus_tokenized]

        ## generate idx
        logger.info("generate idx test")

        if use_bert_feature_matrix:
            idx_frozen = pd.read_csv(data_path + "/" + str(input_file_name) + "_" + str(max_length_of_document_vector_bert) + "_bert_feature_matrix.csv",
                                     delimiter=",", header=None).values.tolist()

            idx = [idx_frozen[i] for i in X_test_ids]

        else:
            idx = [tokenizer.encode(seq.split(" "), is_split_into_words=True) for seq in txt2seq]
            minlen = min([len(i) for i in idx])
            idx = [i[:max_length_of_document_vector_bert] for i in idx]

            if save_bert_feature_matrix:
                np.savetxt(data_path + "/" + str(input_file_name) + "_" + str(max_length_of_document_vector_bert) + "_bert_feature_matrix.csv",
                           idx,
                           delimiter=",",
                           fmt='% s')

        ## feature matrix
        logger.info("feature matrix test")

        if small_model:
            # create X_test for DistilBert
            X_test = [np.array(idx, dtype='int32'), np.array(masks, dtype='int32')]

        else:
            # create X_test for Bert
            ## generate segments
            logger.info("generate segments")
            segments = []
            for seq in txt2seq:
                temp, i = [], 0
                for token in seq.split(" "):
                    temp.append(i)
                    if token == "[SEP]":
                        i += 1
                segments.append(temp)

            X_test = [np.array(idx, dtype='int32'), np.array(masks, dtype='int32'), np.array(segments, dtype='int32')]

        # apply model
        logger.info("APPLY MODEL")
        predicted_prob = model_loaded.predict(X_test)
        predicted = [dic_y_mapping[np.argmax(pred)] for pred in predicted_prob]
        predicted_bin = np.array([np.argmax(pred) for pred in predicted_prob])

    now = time.asctime()

    ###save results to file
    logger.info("SAVE RESULTS")

    try:
        journal = dtf[journal_field].str.lower().tolist()
    except:
        journal = [None for i in range(len(dtf))]
    try:
        pubyear = dtf["Publication Year"].tolist()
    except:
        pubyear = [None for i in range(len(dtf))]
    try:
        WOS_ID = dtf["UT (Unique WOS ID)"].tolist()
    except:
        WOS_ID = [None for i in range(len(dtf))]
    try:
        author = dtf["Author Full Names"].tolist()
    except:
        author = [None for i in range(len(dtf))]
    try:
        title = dtf["Article Title"].tolist()
    except:
        title = [None for i in range(len(dtf))]
    try:
        abstract = dtf["Abstract"].tolist()
    except:
        abstract = [None for i in range(len(dtf))]
    try:
        times_cited = dtf["Times Cited, All Databases"].tolist()
    except:
        times_cited = [None for i in range(len(dtf))]
    try:
        abstract = dtf[text_field].tolist()
    except:
        try:
            abstract = dtf["Abstract"].tolist()
        except:
            abstract = [None for i in range(len(dtf))]
    try:
        WOS_category = dtf["WoS Categories"].tolist()
    except:
        WOS_category = [None for i in range(len(dtf))]
    try:
        research_area = dtf["Research Areas"].tolist()
    except:
        research_area = [None for i in range(len(dtf))]
    try:
        label = dtf[label_field].tolist()
    except:
        try:
            label = dtf["labels"].tolist()
        except:
            label = [None for i in range(len(dtf))]
    try:
        keywords_author = dtf["Author Keywords"].tolist()
    except:
        keywords_author = [None for i in range(len(dtf))]
    try:
        keywords_plus = dtf["Keywords Plus"].tolist()
    except:
        keywords_plus = [None for i in range(len(dtf))]

    results_df = pd.DataFrame({"time": [now for i in predicted],
                               "input_data": [input_file_name for i in predicted],
                               "model_file_name": [model_file_name for i in predicted],
                               "journal": journal,
                               "pubyear": pubyear,
                               "author": author,
                               "times_cited": times_cited,
                               "predicted": predicted,
                               "predicted_bin": predicted_bin,
                               "predicted_prob": predicted_prob[:, 1],
                               "label": label,
                               "title": title,
                               "abstract": abstract,
                               "research_area": research_area,
                               "WOS_category": WOS_category,
                               "WOS_ID": WOS_ID,
                               "keywords_author": keywords_author,
                               "keywords_plus": keywords_plus})

    results_path = data_path + "/results/" + results_file_name + ".csv"

    if save_results:
        results_df.to_csv(results_path, index=False)

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

'''
#######################################
SECTION 2
#######################################

Train a new model on a labeled dataset
-------------------------------------------------------------------
'''
if not use_model:
    ##print parameters to console
    logger.info(parameters)

    if tfidf:
        logger.info(parameters_tfidf)

    if w2v:
        logger.info(parameters_w2v)

    if bert:
        logger.info(parameters_bert)

    ## load utils
    from Utils import utils_ortho_hetero as fcts




    ##load data
    logger.info("LOAD DATA")

    dtf = pd.read_csv(data_path + "/" + input_file_name + ".csv")
    dtf["index"] = np.arange(dtf.shape[0])

    ##plot label distribution
    if plot == 1 or plot == 2:
        logger.info("plot label distribution")
        fig, ax = plt.subplots()
        fig.suptitle("Label Distribution in Original Data", fontsize=12)
        dtf[label_field].reset_index().groupby(label_field).count().sort_values(by="index").plot(kind="barh", legend=False, ax=ax).grid(axis='x')
        plt.show()


    ### TRAIN TEST SPLIT
    logger.info("TRAIN TEST SPLIT")

    dtf["index"] = np.arange(dtf.shape[0])

    if journal_split:
        ## save file with journals which will be used as test-set
        all_journals = dtf[[journal_field, label_field]].drop_duplicates().copy()
        all_journals[journal_field] = all_journals[journal_field].str.lower().copy()
        all_journals = all_journals.drop_duplicates().copy()


        if journal_list != False:
            ## select journals from list if specified
            try:
                all_journals = all_journals.iloc[journal_list] #if index is provided
            except:
                all_journals = all_journals[all_journals[journal_field].isin(journal_list)] #if names are provided

    else:
        ## if no journal split: set dummy variable
        all_journals = pd.DataFrame({journal_field: ["random"], label_field: ["random"]})
        test_journal = None
        test_label = None

    ## only relevant if journal_split == True:
    ## all journals up to "last journal" will be used as test-set
    if num_journals == "all":
        last_journal = len(all_journals)
    else:
        last_journal = num_journals

    loop_number = 0

    ## cross validation loop through all journals
    for index, all_test in all_journals.iterrows():
        loop_number = loop_number + 1

        ##break from loop if last journal is reached
        if loop_number > last_journal:
            break

        logger.info("Loop Nr. = " + str(loop_number))

        logger.info("TRAIN TEST SPLIT")

        ## use full data-set for training and testing (to train full model after model has been selected)
        if train_on_all:
            dtf_train = dtf.copy()
            dtf_test = dtf_train.copy()

        else:
            ## use articles from one journal as test-set at a time (loop through all journals)
            if journal_split == True:

                test_label = all_test[label_field]

                test_journal = all_test[journal_field]

                logger.info("Journal = " + str(test_journal))
                logger.info("Label = " + str(test_label))

                dtf_train = dtf.loc[dtf[journal_field].str.lower() != test_journal].copy()
                dtf_test = dtf.loc[dtf[journal_field].str.lower() == test_journal].copy()

                training_set_id = ''.join(test_journal.split()) + str(int(time.time() * 1000))


            else:
                ## use reproducible train-test split
                if use_reproducible_train_test_split:
                    dtf_train = pd.read_csv(data_path + "/" + train_set_name + ".csv")
                    dtf_test = pd.read_csv(data_path + "/" + test_set_name + ".csv")
                    training_set_id = "use_reproducible_train_test_split"
                    dtf_train["index"] = np.arange(dtf_train.shape[0])
                    dtf_test["index"] = np.arange(dtf_test.shape[0])

                ## create net train test split
                else:
                    dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=test_size, random_state=42)
                    training_set_id = "random" + str(int(time.time() * 1000))




        # balance dataset
        logger.info("BALANCE TRAINING SET")

        # apply random oversampling to training set
        if training_set == "oversample":
            over_sampler = RandomOverSampler(random_state=42)
            X_train, y_train = over_sampler.fit_resample(pd.DataFrame({"X": dtf_train[text_field_clean], "X_raw": dtf_train[text_field], "ID": dtf_train["index"]}), pd.DataFrame({"y": dtf_train[label_field]}))
            X_train_ids = X_train["ID"].tolist()
            X_train_raw = pd.DataFrame({"X": X_train["X_raw"].tolist()})
            X_train = pd.DataFrame({"X": X_train["X"].tolist()})

        # apply random undersampling to training set
        elif training_set == "undersample":
            under_sampler = RandomUnderSampler(random_state=42)
            X_train, y_train = under_sampler.fit_resample(pd.DataFrame({"X": dtf_train[text_field_clean], "X_raw": dtf_train[text_field], "ID": dtf_train["index"]}), pd.DataFrame({"y": dtf_train[label_field]}))
            X_train_ids = X_train["ID"].tolist()
            X_train_raw = pd.DataFrame({"X": X_train["X_raw"].tolist()})
            X_train = pd.DataFrame({"X": X_train["X"].tolist()})

        # apply no balancing to training set
        else:
            X_train = pd.DataFrame({"X": dtf_train[text_field_clean], "X_raw": dtf_train[text_field], "ID": dtf_train["index"]})
            X_train_ids = X_train["ID"].tolist()
            y_train = pd.DataFrame({"y": dtf_train[label_field]})
            X_train_raw = pd.DataFrame({"X": X_train["X_raw"].tolist()})
            X_train = pd.DataFrame({"X": X_train["X"].tolist()})

        # save training set to series for further processing
        X_train_series = X_train.squeeze(axis=1)
        X_train_raw_series = X_train_raw.squeeze(axis=1)
        y_train_series = y_train.squeeze(axis=1)

        ## plot label distributions on train set to check whether balancing worked
        if plot == 1 or plot == 2:
            fig, ax = plt.subplots()
            fig.suptitle("Label Distribution in Training Data", fontsize=12)
            y_train_series.value_counts().plot(kind="barh", legend=False, ax=ax).grid(axis='x')
            plt.show()


        trainingset_id = "TRAININGSET" + str(int(time.time() * 1000)) #store training-set id to results



        '''
        #######################################
        SECTION 2.1
        #######################################
        TRAIN TFIDF MODEL
        ----------------------------------------------------
        '''
        if current_model == "tfidf":

            # TFIDF
            logger.info("TFIDF")

            loop_tfidf_classifier = 0

            ## loop through list of classifiers
            for tfidf_classifier in tfidf_classifier_list:

                loop_tfidf_classifier += 1

                ## set classifier
                if tfidf_classifier == "naive_bayes":
                    classifier = naive_bayes.MultinomialNB()
                elif tfidf_classifier == "LogisticRegression":
                    classifier = linear_model.LogisticRegression()
                elif tfidf_classifier == "LogisticRegressionCV":
                    classifier = linear_model.LogisticRegressionCV(max_iter=300)
                elif tfidf_classifier == "SVC":
                    classifier = svm.SVC(probability=True)
                elif tfidf_classifier == "RandomForestClassifier":
                    classifier = ensemble.RandomForestClassifier()
                elif tfidf_classifier == "GradientBoostingClassifier":
                    classifier = ensemble.GradientBoostingClassifier()

                ## loop through list of minimum document frequency settings
                loop_min_df = 0
                for min_df in min_df_list:

                    loop_min_df += 1
                    loop_ngram_range = 0

                    ## loop through list of ngram settings
                    for ngram_range in ngram_range_list:

                        loop_ngram_range += 1
                        loop_p_value_limit = 0

                        ## loop through list of p-value limit settings
                        for p_value_limit in p_value_limit_list:
                            loop_p_value_limit += 1

                            logger.info("Loop tfidf_classifier Nr.: " + str(loop_tfidf_classifier))
                            logger.info("tfidf_classifier: " + str(tfidf_classifier))
                            logger.info("Loop min_df Nr: " + str(loop_min_df))
                            logger.info("Loop min_df: " + str(min_df))
                            logger.info("Loop ngram_range Nr: " + str(loop_ngram_range))
                            logger.info("Loop ngram_range: " + str(ngram_range))
                            logger.info("Loop p_value_limit Nr: " + str(loop_p_value_limit))
                            logger.info("Loop p_value_limit: " + str(p_value_limit))

                            class_time_start = time.perf_counter()

                            ## much of the following lines is taken from https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794
                            ## load vectorizer for selected ngram and min_df settings
                            vectorizer = feature_extraction.text.TfidfVectorizer(min_df=min_df, ngram_range=ngram_range)

                            ## load unbalanced data for feature selection
                            X_train_unbalanced = X_train[0:len(dtf_train)]
                            y_train_unbalanced = y_train[0:len(dtf_train)]

                            corpus_unbalanced = X_train_unbalanced["X"]

                            ## fit vectorizer to unbalanced data and create mapping
                            vectorizer.fit(corpus_unbalanced)
                            X_train_new_unbalanced = vectorizer.transform(corpus_unbalanced)
                            dic_vocabulary = vectorizer.vocabulary_

                            # FEATURE SELECTION
                            logger.info("FEATURE SELECTION")
                            y = y_train_unbalanced
                            X_names_unbalanced = vectorizer.get_feature_names()

                            ## select feature based on p-value limit in the chi2 test for correlation with the label
                            dtf_features = pd.DataFrame()
                            for cat in np.unique(y):
                                logger.info("cat: " + str(cat))
                                chi2, p = feature_selection.chi2(X_train_new_unbalanced, y == cat)
                                dtf_features = dtf_features.append(pd.DataFrame({"feature": X_names_unbalanced, "score": 1 - p, "y": cat}))
                                dtf_features = dtf_features.sort_values([label_field, "score"], ascending=[True, False])
                                dtf_features = dtf_features[dtf_features["score"] > p_value_limit]

                            X_names_new_unbalanced = dtf_features["feature"].unique().tolist()

                            # shorter
                            logger.info("SHORTENING VOCABULARY")

                            ## create and fit new vectorizer based on the selected features only
                            if len(X_names_new_unbalanced) > 0:
                                vectorizer_new = feature_extraction.text.TfidfVectorizer(vocabulary=X_names_new_unbalanced)
                            else:
                                vectorizer_new = feature_extraction.text.TfidfVectorizer(vocabulary=X_names_unbalanced)
                                p_value_limit = "no limit"

                            corpus = X_train["X"]
                            vectorizer_new.fit(corpus)
                            X_train_new2 = vectorizer_new.transform(corpus)
                            dic_vocabulary = vectorizer_new.vocabulary_

                            # classify
                            logger.info("SET UP CLASSIFIER")

                            ## pipeline
                            model = pipeline.Pipeline([("vectorizer_new", vectorizer_new),
                                                       ("classifier", classifier)])

                            ## train classifier
                            logger.info("TRAIN CLASSIFIER")

                            y_train_new = y_train.values.ravel()

                            model["classifier"].fit(X_train_new2, y_train_new)

                            if save_model:
                                model_file_path = (data_path + "/models/" + model_file_name + ".pkl")
                                with open(model_file_path, 'wb') as file:
                                    pickle.dump(model, file)


                            ## encode y
                            logger.info("encode y")
                            dic_y_mapping = {n: label for n, label in enumerate(np.unique(y_train_new))}
                            inverse_dic = {v: k for k, v in dic_y_mapping.items()}
                            y_train_bin = np.array([inverse_dic[y] for y in y_train_new])

                            ## test
                            logger.info("TEST CLASSIFIER")

                            X_test = dtf_test[text_field_clean].values
                            y_test = dtf_test[label_field].values.ravel()

                            predicted = model.predict(X_test)
                            predicted_prob = model.predict_proba(X_test)

                            logger.info("TEST CALC")
                            predicted_bin = np.array([np.argmax(pred) for pred in predicted_prob])

                            y_test_bin = np.array([inverse_dic[y] for y in y_test])
                            classes = np.array([dic_y_mapping[0], dic_y_mapping[1]])


                            class_time_total = time.perf_counter() - class_time_start
                            logger.info(f"classification with {min_df} features and ngram_range {ngram_range} for {len(dtf)} samples in {class_time_total} seconds")

                            # results general
                            now = time.asctime()
                            result_id = "RESULT" + str(int(time.time() * 1000))

                            result_all = pd.DataFrame({"time": [now],
                                                       "trainingset_id": [trainingset_id],
                                                       "result_id": [result_id],
                                                       "length_data": [len(dtf)],
                                                       "length_training_orig": [len(dtf_train)],
                                                       "length_training_samp": [len(X_train_raw_series)],
                                                       "test_journal": [test_journal],
                                                       "test_label": [test_label],
                                                       "use_reproducible_train_test_split": [use_reproducible_train_test_split],
                                                       "train_set_name": [train_set_name],
                                                       "test_set_name": [test_set_name],
                                                       "tfidf": [tfidf],
                                                       "w2v": [w2v],
                                                       "bert": [bert],
                                                       "duration": [class_time_total],
                                                       "current_model": [current_model]})

                            # results tfidf
                            result_tfidf = pd.DataFrame({"min_df": [min_df],
                                                         "p_value_limit": [p_value_limit],
                                                         "ngram_range": [ngram_range],
                                                         "tfidf_classifier": [tfidf_classifier],
                                                         "number_relevant_features": [len(X_names_new_unbalanced)]})

                            ## test
                            y_test = dtf_test[label_field].values

                            ### EVALUATION
                            result_fct = fcts.evaluate(classes=classes,
                                                       y_test_bin=y_test_bin,
                                                       predicted_bin=predicted_bin,
                                                       predicted_prob=predicted_prob)

                            result = pd.concat([result_all, result_fct, result_tfidf], axis=1)

                            logger.info("RESULT DETAILS:")
                            logger.info(result)

                            ## save the results to file for evaluation
                            if save_results:
                                logger.info("SAVING RESULTS")

                                results, pred_prob_df = fcts.save_results(data_path=data_path,
                                                                          results_file_name=results_file_name,
                                                                          result=result,
                                                                          dtf_test=dtf_test,
                                                                          trainingset_id=trainingset_id,
                                                                          result_id=result_id,
                                                                          predicted_prob=predicted_prob,
                                                                          y_test_bin=y_test_bin)

                            ## save the weights to file for interpretation
                            if save_weights:
                                words = vectorizer_new.get_feature_names()
                                weights = [i for i in model["classifier"].coef_[0]]
                                dtf_weights = pd.DataFrame({"words": words, "weights": weights})

                                weights_loaded_dict = {k: v for k, v in zip(dtf_weights["words"], dtf_weights["weights"])}

                                corpus = dtf_test[text_field_clean]
                                dtf_weights = pd.DataFrame()
                                for prediction in range(2):
                                    idx = [idx for idx, i in enumerate(predicted_bin) if i == prediction]
                                    corpus_loop = corpus.iloc[idx]
                                    ## create list of n-grams
                                    lst_corpus = []
                                    for string in corpus_loop:
                                        lst_words = string.split()
                                        lst_grams = [" ".join(lst_words[i:i + 1]) for i in range(0, len(lst_words), 1)]
                                        lst_corpus.append(lst_grams)

                                    dtf_corpus = pd.DataFrame({"words": [i for l in lst_corpus for i in l]})
                                    dtf_corpus = dtf_corpus["words"].value_counts()


                                    def catch(dict, w):
                                        try:
                                            return dict[w]
                                        except:
                                            return None


                                    weights_list = [catch(weights_loaded_dict, w) for w in dtf_corpus.index.tolist()]
                                    dtf_weights = dtf_weights.append(pd.DataFrame({"words": dtf_corpus.index.tolist(),
                                                                                   "prediction": [prediction for i in weights_list],
                                                                                   "weights": weights_list,
                                                                                   "counts": dtf_corpus.values.tolist()})).copy()

                                dtf_weights["weighted_counts"] = dtf_weights["counts"] * dtf_weights["weights"]
                                dtf_weights.to_csv(data_path + "/weights/" + model_file_name + "_tfidf_weights.csv", index=False)



        '''
        #######################################
        SECTION 2.2
        #######################################
        
        TRAIN W2V MODEL
        ----------------------------------------------------
        '''
        if current_model == "w2v":
            gigaword_loaded = False # initialize variable

            ## balance embedding set if required
            if embedding_set == "oversample":
                over_sampler = RandomOverSampler(random_state=42)
                X_embed, y_embed = over_sampler.fit_resample(pd.DataFrame({"X": dtf_train[text_field_clean]}), pd.DataFrame({"y": dtf_train[label_field]}))

            else:
                if embedding_set == "undersample":
                    under_sampler = RandomUnderSampler(random_state=42)
                    X_embed, y_embed = under_sampler.fit_resample(pd.DataFrame({"X": dtf_train[text_field_clean]}), pd.DataFrame({"y": dtf_train[label_field]}))


                #IMPLEMENTATION TO FIT SEPARATE EMBEDDINGS FOR ORTHODOX AND HETERODOX DATA --> NOT TESTED
                # elif embedding_set == "heterodox":
                #    X_embed = pd.DataFrame({"X": dtf_train.loc[dtf_train[label_field] == "heterodox"][text_field_clean]})
                #    y_embed = pd.DataFrame({"y": dtf_train.loc[dtf_train[label_field] == "heterodox"][label_field]})

                #elif embedding_set == "samequality":
                #    X_embed = pd.DataFrame({"X": dtf_train.loc[dtf_train[label_field] == "samequality"][text_field_clean]})
                #    y_embed = pd.DataFrame({"y": dtf_train.loc[dtf_train[label_field] == "samequality"][label_field]})


                else:
                    X_embed = pd.DataFrame({"X": dtf_train[text_field_clean]})
                    y_embed = pd.DataFrame({"y": dtf_train[label_field]})


            X_embed_series = X_embed.squeeze(axis=1)
            y_embed_series = y_embed.squeeze(axis=1)

            ## plot embedding data to see whether balancing worked
            if plot == 1 or plot == 2:
                fig, ax = plt.subplots()
                fig.suptitle("Label Distribution in Embedding Data", fontsize=12)
                y_embed_series.value_counts().plot(kind="barh", legend=False, ax=ax).grid(axis='x')
                plt.show()

            loop_num_epochs_for_embedding = 0

            ## loop through num epochs settings
            for num_epochs_for_embedding in num_epochs_for_embedding_list:

                loop_num_epochs_for_embedding += 1
                loop_embedding_vector_length = 0

                ## loop through vector length settings
                for embedding_vector_length in embedding_vector_length_list:

                    loop_embedding_vector_length += 1
                    loop_window_size = 0

                    ## loop through window sizes
                    for window_size in window_size_list:

                        loop_window_size += 1

                        logger.info("Loop num_epochs_for_embedding Nr.: " + str(loop_num_epochs_for_embedding))
                        logger.info("num_epochs_for_embedding: " + str(num_epochs_for_embedding))
                        logger.info("Loop embedding_vector_length Nr.: " + str(loop_num_epochs_for_embedding))
                        logger.info("embedding_vector_length: " + str(embedding_vector_length))
                        logger.info("Loop window_size Nr.: " + str(loop_window_size))
                        logger.info("window_size: " + str(window_size))

                        ## much of the following lines is taken from https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794
                        # FEATURE ENGINEERING

                        logger.info("FEATURE ENGINEERING")

                        logger.info("FEATURE ENGINEERING FOR TRAINING SET")

                        corpus = X_embed_series

                        ## create list of lists of unigrams
                        lst_corpus = []
                        for string in corpus:
                            lst_words = string.split()
                            lst_grams = [' '.join(lst_words[i:i + 1]) for i in range(0, len(lst_words), 1)]
                            lst_corpus.append(lst_grams)

                        ## detect bigrams and trigrams
                        bigrams_detector = gensim.models.phrases.Phrases(lst_corpus, delimiter=' ', min_count=5, threshold=10)
                        bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
                        trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[lst_corpus], delimiter=" ", min_count=2, threshold=10)
                        trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)

                        logger.info("STARTING WORD EMBEDDING")

                        ##if pre trained embeddings should be loaded from gensim
                        if use_gigaword:
                            if gigaword_loaded != True:
                                gigaword_start = time.perf_counter()

                                ##define model name to set results file name
                                modelname_raw = "glove-wiki-gigaword-" + str(embedding_vector_length)

                                if journal_split:
                                    modelname = modelname_raw + "_word2vec_" + str(test_journal.replace(" ", "_")) + "_numabs_" + str(len(dtf))
                                else:
                                    modelname = modelname_raw + "_numabs_" + str(len(dtf))

                                pretrained_vectors = modelname_raw

                                ##load pre-trained vectors
                                nlp = gensim_api.load(pretrained_vectors)


                                ## set to true to break from loop
                                gigaword_loaded = True

                                ## output time required to load gigaword
                                gigaword_end = time.perf_counter()
                                gigaword_time = gigaword_end - gigaword_start
                                logger.info(f"loading gigaword vectors in {gigaword_time} seconds")


                        ##if pre trained embeddings should be loaded (self trained embeddings
                        if use_embeddings:
                            load_embeddings_start = time.perf_counter()

                            ##set modelname and load embeddings
                            if journal_split:
                                modelname = "word2vec_" + str(test_journal.replace(" ", "_")) + "_numabs_" + str(len(dtf)) + "_embedlen_" + str(embedding_vector_length) + "_embedepo_" + str(num_epochs_for_embedding) + "_window_" + str(window_size) + "_embed_" + str(embedding_set)
                            else:
                                modelname = "word2vec_numabs_" + str(len(dtf)) + "_embedlen_" + str(embedding_vector_length) + "_embedepo_" + str(num_epochs_for_embedding) + "_window_" + str(window_size) + "_embed_" + str(embedding_set)

                            if which_embeddings == False:
                                pretrained_vectors = str(data_path) + "/" + embedding_folder + "/" + modelname
                            else:
                                pretrained_vectors = str(data_path) + "/" + embedding_folder + "/" + which_embeddings

                            nlp = gensim.models.word2vec.Word2Vec.load(pretrained_vectors)


                        ##if new embeddings need to be trained
                        if train_new:
                            train_embeddings_start = time.perf_counter()

                            ##set model name
                            if journal_split:
                                modelname_raw = "word2vec_" + str(test_journal.replace(" ", "_")) + "_numabs_" + str(len(dtf)) + "_embedlen_" + str(embedding_vector_length) + "_embedepo_" + str(num_epochs_for_embedding) + "_window_" + str(window_size) + "_embed_" + str(embedding_set)

                            else:
                                modelname_raw = "word2vec_numabs_" + str(len(dtf)) + "_embedlen_" + str(embedding_vector_length) + "_embedepo_" + str(num_epochs_for_embedding) + "_window_" + str(window_size) + "_embed_" + str(embedding_set)

                            modelname = "newembedding_" + str(modelname_raw)

                            ## train word2vec module with required settings
                            nlp = gensim.models.word2vec.Word2Vec(lst_corpus, vector_size=embedding_vector_length, window=window_size, sg=1, epochs=num_epochs_for_embedding, workers=cores)

                            ## save word2vec embeddings for further use
                            nlp.save(str(data_path) + "/" + embedding_folder + "/" + modelname_raw)

                            train_embeddings_end = time.perf_counter()
                            train_embeddings_time = train_embeddings_end - train_embeddings_start
                            logger.info(f"training word2vec for {len(dtf)} documents and {num_epochs_for_embedding} epochs in {train_embeddings_time} seconds")

                        logger.info("WORD EMBEDDING FINISHED")

                        ## classification with BiLSTM
                        if embedding_only == False:

                            logger.info("START TOKENIZE")

                            ## tokenize text
                            tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ', oov_token="NaN", filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
                            tokenizer.fit_on_texts(lst_corpus)
                            dic_vocabulary = tokenizer.word_index

                            ## tokenize textsave tokenizer
                            tokenizer_file_path = (data_path + "/models/" + model_file_name + "_tokenizer.pkl")
                            with open(tokenizer_file_path, 'wb') as file:
                                pickle.dump(tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)

                            ## create the corpus
                            logger.info("START CREATING CORPUS")

                            corpus = X_train_series

                            ## the below lines are from https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794
                            ## create list of n-grams
                            lst_corpus = []
                            for string in corpus:
                                lst_words = string.split()
                                lst_grams = [" ".join(lst_words[i:i + 1]) for i in range(0, len(lst_words), 1)]
                                lst_corpus.append(lst_grams)

                            ## detect common bigrams and trigrams using the fitted detectors
                            lst_corpus = list(bigrams_detector[lst_corpus])
                            lst_corpus = list(trigrams_detector[lst_corpus])

                            ## create sequence
                            lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)

                            loop_max_length_of_document_vector_w2v = 0

                            ## loop through different choices wot document vector lengths
                            for max_length_of_document_vector_w2v in max_length_of_document_vector_w2v_list:

                                loop_max_length_of_document_vector_w2v += 1
                                logger.info("Loop max_length_of_document_vector_w2v Nr: " + str(loop_max_length_of_document_vector_w2v))
                                logger.info("max_length_of_document_vector_w2v: " + str(max_length_of_document_vector_w2v))

                                class_time_start = time.perf_counter()

                                ## create feature matrix

                                ## padding sequence
                                X_train_new = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=max_length_of_document_vector_w2v, padding="post", truncating="post")

                                logger.info("SET UP EMBEDDING MATRIX")

                                ## start the matrix (length of vocabulary x vector size) with all 0s
                                embeddings = np.zeros((len(dic_vocabulary) + 1, embedding_vector_length))

                                if use_gigaword:
                                    for word, idx in dic_vocabulary.items():
                                        ## update the row with vector
                                        try:
                                            embeddings[idx] = nlp[word]
                                        ## if word not in model then skip and the row stays all 0s
                                        except:
                                            pass

                                else:
                                    for word, idx in dic_vocabulary.items():
                                        ## update the row with vector
                                        try:
                                            embeddings[idx] = nlp.wv[word]
                                        ## if word not in model then skip and the row stays all 0s
                                        except:
                                            pass

                                loop_classifier_loss_function_w2v = 0

                                ## loop through different loss functions
                                for classifier_loss_function_w2v in classifier_loss_function_w2v_list:
                                    loop_classifier_loss_function_w2v += 1
                                    logger.info("Loop classifier_loss_function_w2v Nr: " + str(loop_classifier_loss_function_w2v))
                                    logger.info("classifier_loss_function_w2v: " + str(classifier_loss_function_w2v))

                                    ## build model architecture
                                    logger.info("NETWORK ARCHITECTURE")


                                    ## code attention layer (not required, but can be used for interpretation)
                                    def attention_layer(inputs, neurons):
                                        x = layers.Permute((2, 1))(inputs)
                                        x = layers.Dense(neurons, activation="softmax")(x)
                                        x = layers.Permute((2, 1), name="attention")(x)
                                        x = layers.multiply([inputs, x])
                                        return x

                                    ## input
                                    x_in = layers.Input(shape=(max_length_of_document_vector_w2v,))

                                    ## embedding
                                    x = layers.Embedding(input_dim=embeddings.shape[0],
                                                         output_dim=embeddings.shape[1],
                                                         weights=[embeddings],
                                                         input_length=max_length_of_document_vector_w2v, trainable=False)(x_in)

                                    ## apply attention
                                    x = attention_layer(x, neurons=max_length_of_document_vector_w2v)

                                    ## 2 layers of bidirectional lstm
                                    x = layers.Bidirectional(layers.LSTM(units=max_length_of_document_vector_w2v, dropout=0.2, return_sequences=True))(x)
                                    x = layers.Bidirectional(layers.LSTM(units=max_length_of_document_vector_w2v, dropout=0.2))(x)

                                    ## final dense layers
                                    x = layers.Dense(64, activation='relu')(x)
                                    y_out = layers.Dense(2, activation='softmax')(x)

                                    ## compile
                                    model = models.Model(x_in, y_out)
                                    model.compile(loss=classifier_loss_function_w2v, optimizer='adam', metrics=['accuracy'])

                                    ##check model architecture
                                    model.summary()

                                    logger.info("ENCODING FEATURES")

                                    ## encode y
                                    dic_y_mapping = {n: label for n, label in enumerate(np.unique(y_train))}
                                    inverse_dic = {v: k for k, v in dic_y_mapping.items()}
                                    y_train_bin = np.array([inverse_dic[y] for y in y_train['y']])

                                    ## loop through choices for nr. of epochs
                                    loop_num_epochs_for_classification = 0
                                    for num_epochs_for_classification in num_epochs_for_classification_list:

                                        loop_num_epochs_for_classification += 1
                                        loop_w2v_batch_size = 0

                                        ## loop through choiced forbatch size
                                        for w2v_batch_size in w2v_batch_size_list:

                                            loop_w2v_batch_size += 1

                                            logger.info("Loop num_epochs_for_classification Nr: " + str(loop_num_epochs_for_classification))
                                            logger.info("num_epochs_for_classification: " + str(num_epochs_for_classification))
                                            logger.info("Loop w2v_batch_size Nr: " + str(loop_w2v_batch_size))
                                            logger.info("w2v_batch_size: " + str(w2v_batch_size))

                                            logger.info("TRAINING")

                                            ## train
                                            train_start = time.perf_counter()
                                            model.fit(x=X_train_new, y=y_train_bin, batch_size=w2v_batch_size, epochs=num_epochs_for_classification, shuffle=True, verbose=0, validation_split=0.3, workers=cores)

                                            ## save model
                                            logger.info("SAVING")
                                            if save_model:
                                                model_file_path = (data_path + "/models/" + model_file_name)
                                                model.save(model_file_path)

                                            class_time_total = time.perf_counter() - class_time_start
                                            logger.info(f"classification with {num_epochs_for_classification} epochs batch size {w2v_batch_size} for {len(dtf)} samples in {class_time_total} seconds")

                                            # results GENERAL
                                            now = time.asctime()
                                            result_id = "RESULT" + str(int(time.time() * 1000))

                                            result_all = pd.DataFrame({"time": [now],
                                                                       "trainingset_id": [trainingset_id],
                                                                       "result_id": [result_id],
                                                                       "length_data": [len(dtf)],
                                                                       "length_training_orig": [len(dtf_train)],
                                                                       "length_training_samp": [len(X_train_raw_series)],
                                                                       "test_journal": [test_journal],
                                                                       "test_label": [test_label],
                                                                       "use_reproducible_train_test_split": [use_reproducible_train_test_split],
                                                                       "train_set_name": [train_set_name],
                                                                       "test_set_name": [test_set_name],
                                                                       "tfidf": [tfidf],
                                                                       "w2v": [w2v],
                                                                       "bert": [bert],
                                                                       "duration": [class_time_total],
                                                                       "current_model": [current_model]})

                                            # results w2v
                                            result_w2v = pd.DataFrame({"use_gigaword": [use_gigaword],
                                                                       "use_embeddings": [use_embeddings],
                                                                       "embedding_folder": [embedding_folder],
                                                                       "train_new": [train_new],
                                                                       "embedding_vector_length": [embedding_vector_length],
                                                                       "num_epochs_for_embedding": [num_epochs_for_embedding],
                                                                       "window_size": [window_size],
                                                                       "embedding_only": [embedding_only],
                                                                       "embedding_set": [embedding_set],
                                                                       "max_length_of_document_vector_w2v": [max_length_of_document_vector_w2v],
                                                                       "classifier_loss_function_w2v": [classifier_loss_function_w2v],
                                                                       "w2v_batch_size": [w2v_batch_size],
                                                                       "num_epochs_for_classification": [num_epochs_for_classification]})

                                            logger.info("FEATURE ENGINEERING FOR TEST SET")

                                            corpus = dtf_test[text_field_clean]

                                            ## create list of n-grams
                                            lst_corpus = []
                                            for string in corpus:
                                                lst_words = string.split()
                                                lst_grams = [" ".join(lst_words[i:i + 1]) for i in range(0, len(lst_words), 1)]
                                                lst_corpus.append(lst_grams)

                                            ## detect common bigrams and trigrams using the fitted detectors
                                            lst_corpus = list(bigrams_detector[lst_corpus])
                                            lst_corpus = list(trigrams_detector[lst_corpus])

                                            ## text to sequence with the fitted tokenizer
                                            lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)

                                            ## padding sequence
                                            X_test = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=max_length_of_document_vector_w2v, padding="post", truncating="post")

                                            ## test
                                            predicted_prob = model.predict(X_test)
                                            predicted = [dic_y_mapping[np.argmax(pred)] for pred in predicted_prob]
                                            y_test = dtf_test[label_field].values
                                            predicted_bin = np.array([np.argmax(pred) for pred in predicted_prob])
                                            y_test_bin = np.array([inverse_dic[y] for y in y_test])
                                            classes = np.array([dic_y_mapping[0], dic_y_mapping[1]])

                                            ### EVALUATION
                                            result_fct = fcts.evaluate(classes=classes,
                                                                       y_test_bin=y_test_bin,
                                                                       predicted_bin=predicted_bin,
                                                                       predicted_prob=predicted_prob)

                                            result = pd.concat([result_all, result_fct, result_w2v], axis=1)

                                            logger.info("RESULT DETAILS:")
                                            logger.info(result)

                                            if save_results:
                                                logger.info("SAVING RESULTS")

                                                results, pred_prob_df = fcts.save_results(data_path=data_path,
                                                                                          results_file_name=results_file_name,
                                                                                          result=result,
                                                                                          dtf_test=dtf_test,
                                                                                          trainingset_id=trainingset_id,
                                                                                          result_id=result_id,
                                                                                          predicted_prob=predicted_prob,
                                                                                          y_test_bin=y_test_bin)



        '''
        #######################################
        SECTION 2.3
        #######################################

        TRAIN BERT MODEL
        ----------------------------------------------------
        '''
        if current_model == "bert":

            small_model_loop = 0

            ## loop through list of "distilbert", and "bert"
            for small_model in small_model_list:

                small_model_loop += 1

                ## tokenizer
                if small_model:
                    ## load distilbert tokenizer
                    try:
                        logger.info("LOAD DISTILBERT TOKENIZER FROM FOLDER")
                        tokenizer = transformers.AutoTokenizer.from_pretrained(data_path + '/distilbert-base-uncased/', do_lower_case=True)

                    except:
                        logger.info("LOAD DISTILBERT TOKENIZER FROM API")
                        tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

                else:
                    ## load bert tokenizer
                    try:
                        logger.info("LOAD BERT TOKENIZER FROM FOLDER")
                        tokenizer = transformers.AutoTokenizer.from_pretrained(data_path + '/bert-base-uncased/', do_lower_case=True)
                    except:
                        logger.info("LOAD BERT TOKENIZER FROM API")
                        tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


                max_length_of_document_vector_bert_loop = 0

                ## loop through choices of maximum document length
                for max_length_of_document_vector_bert in max_length_of_document_vector_bert_list:

                    max_length_of_document_vector_bert_loop += 1
                    logger.info("Loop small_model Nr: " + str(small_model_loop))
                    logger.info("small_model : " + str(small_model))
                    logger.info("Loop max_length_of_document_vector_bert Nr: " + str(max_length_of_document_vector_bert_loop))
                    logger.info("max_length_of_document_vector_bert : " + str(max_length_of_document_vector_bert))

                    ## pre-processing specific for transformers (w/out stopword removal and lemmatization)
                    text_lst = [text[:-50] for text in X_train_raw["X"]]
                    text_lst = [' '.join(text.split()[:max_length_of_document_vector_bert]) for text in text_lst]

                    subtitles = ["design", "methodology", "approach", "originality", "value", "limitations", "implications", "elsevier", "purpose"]

                    text_lst = [word for word in text_lst if word not in subtitles]

                    corpus = text_lst

                    ### LINES BELOW ARE FROM https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794
                    ## Feature engineering train set
                    logger.info("Fearute engineering train set")

                    ## add special tokens
                    logger.info("add special tokens")

                    maxqnans = np.int((max_length_of_document_vector_bert - 5))  # / 2)
                    corpus_tokenized = ["[CLS] " +
                                        " ".join(tokenizer.tokenize(re.sub(r'[^\w\s]+|\n', '', str(txt).lower().strip()))[:maxqnans]) +
                                        " [SEP] " for txt in corpus]

                    ## generate masks
                    logger.info("generate masks")
                    masks = [[1] * len(txt.split(" ")) + [0] * (max_length_of_document_vector_bert - len(txt.split(" "))) for txt in corpus_tokenized]

                    ## padding
                    logger.info("padding")
                    txt2seq = [txt + " [PAD]" * (max_length_of_document_vector_bert - len(txt.split(" "))) if len(txt.split(" ")) != max_length_of_document_vector_bert else txt for txt in corpus_tokenized]

                    ## generate idx
                    logger.info("generate idx")

                    ## load existing feature matrix
                    if use_bert_feature_matrix:
                        idx_frozen = pd.read_csv(data_path + "/" + str(input_file_name) + "_" + str(max_length_of_document_vector_bert) + "_bert_feature_matrix.csv",
                                                 delimiter=",", header=None).values.tolist()

                        idx = [idx_frozen[i - 1] for i in X_train_ids]

                    ## create new feature matrix
                    else:
                        idx = [tokenizer.encode(seq.split(" "), is_split_into_words=True) for seq in txt2seq]
                        minlen = min([len(i) for i in idx])
                        idx = [i[:max_length_of_document_vector_bert] for i in idx]

                        if save_bert_feature_matrix:
                            np.savetxt(data_path + "/" + str(train_set_name) + "_" + str(max_length_of_document_vector_bert) + "_bert_feature_matrix.csv",
                                       idx,
                                       delimiter=",",
                                       fmt='% s')

                    ## feature matrix
                    logger.info("feature matrix")

                    ## no segments required for distilbert
                    if small_model:
                        X_train_new = [np.asarray(idx, dtype='int32'), np.asarray(masks, dtype='int32')]

                    else:

                        ## generate segments
                        logger.info("generate segments")
                        segments = []
                        for seq in txt2seq:
                            temp, i = [], 0
                            for token in seq.split(" "):
                                temp.append(i)
                                if token == "[SEP]":
                                    i += 1
                            segments.append(temp)

                        X_train_new = [np.asarray(idx, dtype='int32'), np.asarray(masks, dtype='int32'), np.asarray(segments, dtype='int32')]


                    ##CLASSIFIER
                    logger.info("CLASSIFIER")

                    classifier_loss_function_bert_loop = 0

                    ## loop through different choives of loss functions
                    for classifier_loss_function_bert in classifier_loss_function_bert_list:
                        classifier_loss_function_bert_loop += 1
                        logger.info("classifier_loss_function_bert_loop Nr = " + str(classifier_loss_function_bert_loop))
                        logger.info("classifier_loss_function_bert = " + str(classifier_loss_function_bert))

                        ## set up architectures for the model of choice
                        if small_model:

                            ##DISTIL-BERT
                            logger.info("DISTIL-BERT MODEL ARCHITECTURE")

                            ## inputs
                            idx = layers.Input((max_length_of_document_vector_bert), dtype="int32", name="input_idx")
                            masks = layers.Input((max_length_of_document_vector_bert), dtype="int32", name="input_masks")
                            # segments = layers.Input((max_length_of_document_vector_bert), dtype="int32", name="input_segments")

                            ## pre-trained bert with config
                            config = transformers.DistilBertConfig(dropout=0.2, attention_dropout=0.2)
                            config.output_hidden_states = False

                            try:
                                nlp = transformers.TFDistilBertModel.from_pretrained(data_path + '/distilbert-base-uncased/', config=config)
                            except:
                                nlp = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased', config=config)

                            bert_out = nlp.distilbert(idx, attention_mask=masks)[0]

                            ## fine-tuning
                            x = layers.GlobalAveragePooling1D()(bert_out)
                            x = layers.Dense(64, activation="relu")(x)
                            y_out = layers.Dense(len(np.unique(y_train)), activation='softmax')(x)

                            ## compile
                            logger.info("compile DISTIL-BERT")
                            model = models.Model([idx, masks], y_out)

                            for layer in model.layers[:3]:
                                layer.trainable = False

                            model.compile(loss=classifier_loss_function_bert, optimizer='adam', metrics=['mse'])

                            model.summary()

                        else:
                            ##BERT
                            logger.info("DISTIL-BERT MODEL ARCHITECTURE")
                            logger.info("BERT")

                            ## inputs
                            idx = layers.Input((max_length_of_document_vector_bert), dtype="int32", name="input_idx")
                            masks = layers.Input((max_length_of_document_vector_bert), dtype="int32", name="input_masks")
                            segments = layers.Input((max_length_of_document_vector_bert), dtype="int32", name="input_segments")

                            ## pre-trained bert
                            try:
                                nlp = transformers.TFBertModel.from_pretrained(data_path + '/bert-base-uncased/')
                            except:
                                nlp = transformers.TFBertModel.from_pretrained('bert-base-uncased')

                            bert_out = nlp.bert([idx, masks, segments])
                            '''
                            sequence_out = bert_out.last_hidden_state
                            pooled_out = bert_out.pooler_output
                            '''
                            seq_out, _ = bert_out[0], bert_out[1]
                            ## fine-tuning
                            x = layers.GlobalAveragePooling1D()(seq_out)
                            x = layers.Dense(64, activation="relu")(x)
                            y_out = layers.Dense(len(np.unique(y_train)), activation='softmax')(x)

                            ## compile
                            logger.info("compile BERT")
                            model = models.Model([idx, masks, segments], y_out)

                            for layer in model.layers[:4]:
                                layer.trainable = False

                            model.compile(loss=classifier_loss_function_bert, optimizer='adam', metrics=['mse'])

                            model.summary()


                        # Feature engineer Test set (incl. transformer specific pre-processing)
                        logger.info("Feature engineer Test set")

                        X_test_ids = dtf_test["index"].tolist()

                        text_lst = [text[:-50] for text in dtf_test[text_field]]
                        text_lst = [' '.join(text.split()[:maxqnans]) for text in text_lst]
                        # text_lst = [text for text in text_lst if text]

                        corpus = text_lst

                        ## add special tokens
                        logger.info("add special tokens test")
                        maxqnans = np.int((max_length_of_document_vector_bert - 5))  # / 2)
                        corpus_tokenized = ["[CLS] " +
                                            " ".join(tokenizer.tokenize(re.sub(r'[^\w\s]+|\n', '', str(txt).lower().strip()))[:maxqnans]) +
                                            " [SEP] " for txt in corpus]

                        ## generate masks
                        logger.info("generate masks test")
                        masks = [[1] * len(txt.split(" ")) + [0] * (max_length_of_document_vector_bert - len(txt.split(" "))) for txt in corpus_tokenized]

                        ## padding
                        logger.info("padding test")
                        txt2seq = [txt + " [PAD]" * (max_length_of_document_vector_bert - len(txt.split(" "))) if len(txt.split(" ")) != max_length_of_document_vector_bert else txt for txt in corpus_tokenized]

                        ## generate idx
                        logger.info("generate idx test")

                        if use_bert_feature_matrix:
                            idx_frozen = pd.read_csv(data_path + "/" + str(input_file_name) + "_" + str(max_length_of_document_vector_bert) + "_bert_feature_matrix.csv",
                                                     delimiter=",", header=None).values.tolist()

                            idx = [idx_frozen[i] for i in X_test_ids]

                        else:
                            idx = [tokenizer.encode(seq.split(" "), is_split_into_words=True) for seq in txt2seq]
                            minlen = min([len(i) for i in idx])
                            idx = [i[:max_length_of_document_vector_bert] for i in idx]

                            if save_bert_feature_matrix:
                                np.savetxt(data_path + "/" + str(input_file_name) + "_" + str(max_length_of_document_vector_bert) + "_bert_feature_matrix.csv",
                                           idx,
                                           delimiter=",",
                                           fmt='% s')

                        ## feature matrix
                        logger.info("feature matrix test")

                        if small_model:
                            X_test = [np.array(idx, dtype='int32'), np.array(masks, dtype='int32')]

                        else:
                            ## generate segments
                            logger.info("generate segments")
                            segments = []
                            for seq in txt2seq:
                                temp, i = [], 0
                                for token in seq.split(" "):
                                    temp.append(i)
                                    if token == "[SEP]":
                                        i += 1
                                segments.append(temp)

                            X_test = [np.array(idx, dtype='int32'), np.array(masks, dtype='int32'), np.array(segments, dtype='int32')]

                        ## encode y
                        logger.info("encode y")
                        dic_y_mapping = {n: label for n, label in enumerate(np.unique(y_train))}
                        inverse_dic = {v: k for k, v in dic_y_mapping.items()}
                        y_train_bin = np.array([inverse_dic[y] for y in y_train["y"]])

                        bert_batch_size_loop = 0

                        ## loop through choices for batch size
                        for bert_batch_size in bert_batch_size_list:

                            bert_batch_size_loop += 1

                            bert_epochs_loop = 0

                            ## loop through choices for number of epochs
                            for bert_epochs in bert_epochs_list:

                                class_time_start = time.perf_counter()

                                bert_epochs_loop += 1

                                logger.info("max_length_of_document_vector_bert_loop Nr: " + str(max_length_of_document_vector_bert_loop))
                                logger.info("max_length_of_document_vector_bert : " + str(max_length_of_document_vector_bert))
                                logger.info("bert_batch_size Nr: " + str(bert_batch_size_loop))
                                logger.info("bert_batch_size : " + str(bert_batch_size))
                                logger.info("bert_epochs_loop Nr: " + str(bert_epochs_loop))
                                logger.info("bert_epochs : " + str(bert_epochs))

                                ## train
                                model.fit(x=X_train_new, y=y_train_bin, batch_size=bert_batch_size, epochs=bert_epochs, shuffle=True, verbose=1, validation_split=0.3)

                                ## save model to disk for later use
                                if save_model:
                                    model_file_path = str(data_path + "/models/" + model_file_name)
                                    saving = model.save(model_file_path)

                                class_time_total = time.perf_counter() - class_time_start
                                logger.info(f"classification with {bert_epochs} epochs batch size {bert_batch_size} for {len(dtf)} samples in {class_time_total} seconds")

                                # results general
                                now = time.asctime()
                                result_id = "RESULT" + str(int(time.time() * 1000))

                                result_all = pd.DataFrame({"time": [now],
                                                           "trainingset_id": [trainingset_id],
                                                           "result_id": [result_id],
                                                           "length_data": [len(dtf)],
                                                           "length_training_orig": [len(dtf_train)],
                                                           "length_training_samp": [len(X_train_raw_series)],
                                                           "test_journal": [test_journal],
                                                           "test_label": [test_label],
                                                           "use_reproducible_train_test_split": [use_reproducible_train_test_split],
                                                           "train_set_name": [train_set_name],
                                                           "test_set_name": [test_set_name],
                                                           "tfidf": [tfidf],
                                                           "w2v": [w2v],
                                                           "bert": [bert],
                                                           "duration": [class_time_total],
                                                           "current_model": [current_model]})

                                # results bert
                                result_bert = pd.DataFrame({"max_length_of_document_vector_bert": [max_length_of_document_vector_bert],
                                                            "classifier_loss_function_bert": [classifier_loss_function_bert],
                                                            "small_model": [small_model],
                                                            "bert_batch_size": [bert_batch_size],
                                                            "bert_epochs": [bert_epochs]})

                                ## test
                                predicted_prob = model.predict(X_test)
                                predicted = [dic_y_mapping[np.argmax(pred)] for pred in predicted_prob]
                                y_test = dtf_test[label_field].values
                                predicted_bin = np.array([np.argmax(pred) for pred in predicted_prob])
                                y_test_bin = np.array([inverse_dic[y] for y in y_test])
                                classes = np.array([dic_y_mapping[0], dic_y_mapping[1]])

                                ### EVALUATION
                                result_fct = fcts.evaluate(classes=classes,
                                                           # y_test = y_test,
                                                           y_test_bin=y_test_bin,
                                                           # predicted = predicted,
                                                           predicted_bin=predicted_bin,
                                                           predicted_prob=predicted_prob)

                                result = pd.concat([result_all, result_fct, result_bert], axis=1)

                                logger.info("RESULT DETAILS:")
                                logger.info(result)

                                if save_results:
                                    logger.info("SAVING RESULTS")

                                    results, pred_prob_df = fcts.save_results(data_path=data_path,
                                                                              results_file_name=results_file_name,
                                                                              result=result,
                                                                              dtf_test=dtf_test,
                                                                              trainingset_id=trainingset_id,
                                                                              result_id=result_id,
                                                                              predicted_prob=predicted_prob,
                                                                              y_test_bin=y_test_bin)

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

#print total runtime to log
toc = time.perf_counter()
logger.info(f"whole script for {len(dtf)} in {toc - tic} seconds")
print("the end")
