'''
This script is part of the Master Thesis of Damian Durrer (SfS ETH Zürich), submission date 16 December 2021.

This script is part of the following project:
- Github: https://github.com/dipfi/econ-classifier
- Euler: /cluster/work/lawecon/Projects/Ash_Durrer/dev/scripts

The data for reproduction can be shared upon request:
- Alternatively for members of the LawEcon group at ETHZ
    -Euler: /cluster/work/lawecon/Projects/Ash_Durrer/dev/data

Much of the code here is based on:
https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794

PREREQUISITS:
- GET DATA FROM THE ABOVE SOURCES OR FROM THE WEB OF SCIENCE

THIS SCRIPT IS USED TO PREPROCESS THE DATA AS DESCRIBED IN THE MASTERS THESIS "THE DIALECTS OF ECONOSPEAK" BY DAMIAN DURRER (2021)

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

## for data
import pandas as pd
pd.set_option('display.max_columns', None)
#######################################

'''
INPUT REQUIRED: SET PARAMETERS
----------------------------------------------------------
'''

############################################
logging_level = logging.INFO
## Choose level of logs to display in console
#### Recommended: logging.INFO
#### Alternative: logging.DEBUG // #logging.WARNING

print_charts_tables = False
## Choose wether to print diagostics charts and tables
#### Recommended: False // no charts and tables printed
#### Alternative: True // print diagnostic charts and tables

input_file_name = "WOS_top5_new"
## Choose input file name to apply preprocessing to. This must be a file containing abstracts and labels of articles and be located in DATA_PATH
#### Recommended: "WOS_top5_new" // top 5 articles ; "WOS_lee_heterodox_und_samequality_new" // full labeled dataset


output_file_name = "WOS_top5_new_preprocessed_2_test"
## Choose output file name

sample_size = "all"
## Choose sample size to sample the input file and only preprocess a number of observations
#### Recommended: "all" // apply preprocessing to full data
#### Alternative: 1000 // (or any number) to create pre-processed file with only n observations

text_field = "Abstract"
## specify where the texts to preprocess are found
#### Recommended: "Abstract" // WOS name for the abstract field
#### Alternative: "Article Title"

label_field = "labels"
## specify the name of the label field (i.e. where we find the labels "0samequality" and "1heterodox"
#### Recommended: "labels"

min_char = 120
## choose the minimum length of the text --> this is important because the last 40 characters are deleted (because they might contain copyright mentions etc)
#### Recommended: 120

cores = mp.cpu_count()
## choose number of cores to use for parallel preprocessing
#### Recommended: mp.cpu_count() // all available cores
#### Alternative: 1 // only one core

save = True
## choose whether to save the results
#### Recommended: True // save results
#### Alternative: False // don't save results
############################################


'''
FINAL SET UP
----------------------------------------------------------
'''


## LEGACY, cont change
input_file_type = "csv"
input_file_size = "all"
## LEGACY, cont change

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
APPLY PRE-PROCESSING
----------------------------------------------------------
'''

def main():
    logger.info("START MAIN")
    logger.info("Logging Level: " + str(logging_level))
    logger.info("Working Directory: " + str(os.getcwd()))
    logger.info("Data Path: " + str(data_path))

    nltk.download('wordnet')
    nltk.download('stopwords')

    dtf = fcts.load_data(data_path=data_path,
                         input_file_type=input_file_type,
                         input_file_size=input_file_size,
                         input_file_name=input_file_name,
                         sample_size=sample_size)

    dtf = fcts.rename_fields(dtf=dtf,
                             text_field=text_field,
                             label_field=label_field)

    dtf = fcts.split_text(dtf=dtf,
                          min_char=min_char)


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
