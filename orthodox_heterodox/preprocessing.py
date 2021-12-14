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

    scripts_path = config['PATH']['scripts_path']
    project_path = config['PATH']['project_path']
    data_path = config['PATH']['data_path']

    sys.path.append(project_path)
    return data_path, scripts_path, project_path

if __name__ == "__main__":
    data_path, scripts_path, project_path = config()


##parallelization
import multiprocessing as mp

##downloading wordlist in main
import nltk

'''
DISCIPLINES SPECIFIC IMPORTS
'''
## for data
import pandas as pd

pd.set_option('display.max_columns', None)
# import collections
# import json
# from scipy import stats

## for plotting
# import matplotlib.pyplot as plt
# import seaborn as sns
# import wordcloud

## for text processing
# import re
# import nltk

## for language detection
import langdetect

## for sentiment
# from textblob import TextBlob

## for ner
# import spacy

## for vectorizer
# from sklearn import feature_extraction, manifold

## for word embedding
# import gensim.downloader as gensim_api

## for topic modeling
# import gensim

############################################
logging_level = logging.INFO  # logging.DEBUG #logging.WARNING
print_charts_tables = True  # False #True
input_file_name = "WOS_top5_new"
input_file_size = "all" #10000 #"all"
input_file_type = "csv"
output_file_name = "WOS_top5_new_preprocessed_2"
sample_size = "all" #"all"  #input_file_size #10000 #"all"
text_field = "Abstract"  # "title" #"abstract"
label_field = "labels"
#remove_last_n = 50 #remove 45 for elsevier copyright
min_char = 120
cores = mp.cpu_count()  #mp.cpu_count()  #2
save = True  # False #True
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

def main():
    logger.info("START MAIN")
    logger.info("Logging Level: " + str(logging_level))
    logger.info("Working Directory: " + str(os.getcwd()))
    logger.info("Data Path: " + str(data_path))

    nltk.download('wordnet')
    nltk.download('stopwords')

    fcts.hello_world() #test

    dtf = fcts.load_data(data_path=data_path,
                         input_file_type=input_file_type,
                         input_file_size=input_file_size,
                         input_file_name=input_file_name,
                         sample_size=sample_size)
    # logger.info("Data Short Head:\n" + str(dtf.head()) + "\n")

    """
    if save == True:
        fcts.save_data_csv(dtf=dtf,
                           data_path=data_path,
                           output_file_name=input_file_name,
                           sample_size=sample_size)
    """

    dtf = fcts.rename_fields(dtf=dtf,
                             text_field=text_field,
                             label_field=label_field)
    # logger.info("Data Subset & Rename Head:\n" + str(dtf.head()) + "\n")

    dtf = fcts.split_text(dtf=dtf,
                          min_char=min_char)
    # logger.info("Data Languages Head:\n" + str(dtf.head()) + "\n")

    '''
    dtf = fcts.language_detection_wrapper(dtf=dtf,
                                          min_char=min_char,
                                          cores=cores,
                                          function=langdetect.detect)
    # logger.info("Data Languages Head:\n" + str(dtf.head()) + "\n")
    '''
    #lst_stopwords = fcts.load_stopwords()

    dtf = fcts.preprocessing_wrapper(dtf=dtf, cores=cores)
    logger.info("Data Languages Head:\n" + str(dtf.head()) + "\n")


    logger.info("END MAIN")
    toc = time.perf_counter()
    logger.info(f"whole script for {sample_size} in {toc - tic} seconds")

    if save == True:
        fcts.save_data_csv(dtf=dtf,
                           data_path=data_path,
                           output_file_name=output_file_name,
                           sample_size=sample_size)

if __name__ == "__main__":
    main()
