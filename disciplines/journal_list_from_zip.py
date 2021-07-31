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
    config.read(os.getcwd() + '/scripts/config.ini')

    data_path = config['PATH']['data_path']
    scripts_path = config['PATH']['scripts_path']
    project_path = config['PATH']['project_path']

    sys.path.append(project_path)
    return data_path, scripts_path, project_path

if __name__ == "__main__":
    data_path, scripts_path, project_path = config()




##parallelization
import multiprocessing as mp


import pandas as pd
pd.set_option('display.max_columns', None)
#import zipfile
#import re
#from bs4 import BeautifulSoup



############################################
logging_level = logging.INFO  # logging.DEBUG #logging.WARNING
cores = mp.cpu_count()  #2
output_file_name = "journal_list_sociology_test"


'''
target_addresses = ['/cluster/work/lawecon/Data/jstor_econ/raw',
                    '/cluster/work/lawecon/Work/dcai/journal_articles/data/jstor/jstor_sociology',
                    '/cluster/work/lawecon/Work/JSTOR_metadata/political_science',
                    '/cluster/work/lawecon/Work/JSTOR_metadata/african_american_studies',
                    '/cluster/work/lawecon/Work/JSTOR_metadata/american_indian_studies',
                    '/cluster/work/lawecon/Work/JSTOR_metadata/criminology',
                    '/cluster/work/lawecon/Work/JSTOR_metadata/law',
                    '/cluster/work/lawecon/Work/JSTOR_metadata/public_policy',
                    '/cluster/work/lawecon/Work/JSTOR_metadata/management',
                    '/cluster/work/lawecon/Work/JSTOR_metadata/history']
'''

target_addresses = ["C:/Users/damdu/kDrive/Master/Masterarbeit/Ash/dev/data/jstor_sociology_small",
                    "C:/Users/damdu/kDrive/Master/Masterarbeit/Ash/dev/data/jstor_sociology_small"]
'''
target_addresses = ['/cluster/work/lawecon/Data/jstor_econ/raw',
                    '/cluster/work/lawecon/Work/dcai/journal_articles/data/jstor/jstor_sociology',
                    '/cluster/work/lawecon/Work/JSTOR_metadata/political_science']
'''

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
                    #handlers=[logging.FileHandler("log.log"),
                    #          logging.StreamHandler()],
                    format=('%(levelname)s | '
                            '%(asctime)s | '
                            '%(filename)s | '
                            '%(funcName)s() | '
                            '%(lineno)d | \t'
                            '%(message)s'))  # , format='%(levelname)s - %(asctime)s - %(message)s - %(name)s')

logger = logging.getLogger()

from scripts.Utils import utils_journal_list as fcts



def main():
    logger.info("START MAIN")
    logger.info("Logging Level: " + str(logging_level))
    logger.info("Working Directory: " + str(os.getcwd()))
    #logger.info("Data Path: " + str(data_path))

    fcts.hello_world() #test

    counts_df = fcts.journal_list_wrapper(target_addresses = target_addresses,
                                     cores = cores)

# In[2]:

    '''
    #counts_df.to_csv(str(data_path + "/journal_list_sociology_test.csv"))
    counts_df.to_csv("C:/Users/damdu/kDrive/Master/Masterarbeit/Ash/dev/data/journal_list_sociology_test2.csv")



    ###
    '''

    fcts.save_data_csv(dtf = counts_df,
                       data_path = data_path,
                       output_file_name = output_file_name)

    logger.info("END MAIN")
    toc = time.perf_counter()
    logger.info(f"whole script in {toc - tic} seconds")


if __name__ == "__main__":
    main()


