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
#import matplotlib as mpl
#mpl.use('TkAgg')
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
input_file_name = "WOS_lee_heterodox_und_samequality_preprocessed_1000"
input_file_size = "all" #10000 #"all"
input_file_type = "csv"
output_file_name = "WOS_lee_heterodox_und_samequality_preprocessed_wip"
sample_size = "all" #input_file_size #10000 #"all"
use_reproducible_train_test_split = True
train_set_name = "WOS_lee_heterodox_und_samequality_preprocessed_train_9"
test_set_name = "WOS_lee_heterodox_und_samequality_preprocessed_test_1"
text_field_clean = "text_clean"  # "title" #"abstract"
text_field = "text"
label_field = "y"
cores = mp.cpu_count()  #mp.cpu_count()  #2
save = False  # False #True
plot = 0 #0 = none, 1 = some, 2 = all

save_results = True

journal_split = True
num_journals = "all" #3 #"all"
random_journals = False
journal_list = [i for i in range(0,30)] #False # [65,1]

test_size = 0.1 #suggestion: 0.1
training_set = "oversample" # "oversample", "undersample", "heterodox", "samequality" ; suggestion: oversample

results_file_name = "results_test_tfidf_short"

#BERT only
bert = False
small_model_list = [True]
bert_batch_size_list = [64]
bert_epochs_list = [6]
max_length_of_document_vector_bert_list = [350] #np.max([len(i.split()) for i in X_train_series]) #np.quantile([len(i.split()) for i in X_train_series], 0.7) ; suggesion: 350
classifier_loss_function_bert_list = ['sparse_categorical_crossentropy'] #, 'mean_squared_error', 'sparse_categorical_crossentropy', "kl_divergence", 'categorical_hinge'
use_bert_feature_matrix = True
save_bert_feature_matrix = False

############################################


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


from Utils import utils_ortho_hetero as fcts

'''
LOAD DATA
'''
logger.info("LOAD DATA")
logger.info("LOAD DATA")
if __name__ == "__main__":
    dtf = fcts.load_data(data_path = data_path,
                          input_file_name = input_file_name,
                          input_file_size = input_file_size,
                          input_file_type = input_file_type,
                          sample_size = "all")

X_train_raw = pd.DataFrame({"X": dtf[text_field].tolist()})


for small_model in small_model_list:


    if small_model:
        tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    max_length_of_document_vector_bert_loop = 0

    for max_length_of_document_vector_bert in max_length_of_document_vector_bert_list:


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

        np.savetxt(data_path + "/" + str(input_file_name) + "_" + str(max_length_of_document_vector_bert) + "_bert_feature_matrix.csv",
                   idx,
                   delimiter=",",
                   fmt='% s')