## set up
import time

tic = time.perf_counter()
import random

random.seed(10)
import logging

##config set up
import configparser
import os

config = configparser.ConfigParser()
config.read(os.getcwd() + '/code/config.ini')
data_path = config['PATH']['data_path']
code_path = config['PATH']['code_path']
project_path = config['PATH']['project']

##parallelization
import multiprocessing as mp

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
input_file_name = "sample_for_damian"
input_file_size = 500
input_file_type = "csv"
sample_size = input_file_size  # input_file_size #10000
text_field = "abstract"  # "title" #"abstract"
label_field = "discipline"
min_char = 120
cores = 10 #mp.cpu_count()
save = True  # False #True
############################################


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

from code.Utils import utils_disciplines as fcts


def main():
    logger.info("START MAIN")
    logger.info("Logging Level: " + str(logging_level))
    logger.info("Working Directors: " + str(os.getcwd()))
    fcts.hello_world()
    logger.info("Data Path: " + str(data_path))
    dtf = fcts.load_data(data_path=data_path,
                         input_file_type=input_file_type,
                         input_file_size=input_file_size,
                         input_file_name=input_file_name,
                         sample_size=sample_size)
    # logger.info("Data Short Head:\n" + str(dtf.head()) + "\n")

    if save == True:
        fcts.save_data_csv(dtf=dtf,
                           data_path=data_path,
                           output_file_name=input_file_name,
                           sample_size=sample_size)

    dtf = fcts.rename_fields(dtf=dtf,
                             text_field=text_field,
                             label_field=label_field)
    # logger.info("Data Subset & Rename Head:\n" + str(dtf.head()) + "\n")
'''
    dtf = fcts.split_text(dtf=dtf,
                          min_char=min_char)
    # logger.info("Data Languages Head:\n" + str(dtf.head()) + "\n")

    dtf = fcts.language_detection_wrapper(dtf=dtf,
                                          min_char=min_char,
                                          cores=cores,
                                          function=langdetect.detect)
    logger.info("Data Languages Head:\n" + str(dtf.head()) + "\n")

    #lst_stopwords = fcts.load_stopwords()

    dtf = fcts.preprocessing_wrapper(dtf=dtf, cores=cores)
    logger.info("Data Languages Head:\n" + str(dtf.head()) + "\n")


    logger.info("END MAIN")
    toc = time.perf_counter()
    logger.info(f"whole script for {sample_size} in {toc - tic} seconds")
'''

if __name__ == "__main__":
    main()
