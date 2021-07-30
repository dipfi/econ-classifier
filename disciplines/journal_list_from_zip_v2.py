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
config.read(os.getcwd() + '/scripts/.config.ini')

data_path = config['PATH']['data_path']
scripts_path = config['PATH']['scripts_path']
project_path = config['PATH']['project_path']


import sys
sys.path.append(project_path)

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

#target_addresses = [str(data_path + "/jstor_sociology_small")]
target_addresses = ["C:/Users/damdu/kDrive/Master/Masterarbeit/Ash/dev/data/jstor_sociology_small"]

############################################


'''
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
'''
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

    '''
    for count, address in enumerate(target_addresses):

        time_start_listing_zipfiles = time.perf_counter()
        os.chdir(address)
        zfs = os.listdir(address)
        toc = time.perf_counter()
        logger.info(f"time for listing zipfiles: {toc - time_start_listing_zipfiles} seconds")

        time_start_processing_zipfiles = time.perf_counter()
        results = fcts.multiprocessing_wrapper(input_data = zfs, cores = cores)
        toc = time.perf_counter()
        logger.info(f"time for processing zipfiles: {toc - time_start_processing_zipfiles} seconds")

        #results = map(process_zipfile, zfs)
        logger.info(address)

        time_start_dropping_duplicates = time.perf_counter()
        results_df = pd.concat(results).drop_duplicates(subset = ['jstor_id'])
        toc = time.perf_counter()
        logger.info(f"time for adropping duplicates: {toc - time_start_dropping_duplicates} seconds")

        results_df['discipline'] = count
        dfs = []
        dfs.append(results_df)

# In[1]:


    primary_df = pd.concat(dfs)

    time_start_aggregation = time.perf_counter()
    counts_df = primary_df.groupby(by=["journal", "discipline"])["year"].value_counts()
    toc = time.perf_counter()
    logger.info(f"time for aggregation: {toc - time_start_aggregation} seconds")
    '''
    ###
    '''
    counts_df = fcts.journal_list_wrapper(target_addresses = target_addresses,
                                     cores = cores)
    '''
# In[2]:

    '''
    #counts_df.to_csv(str(data_path + "/journal_list_sociology_test.csv"))
    counts_df.to_csv("C:/Users/damdu/kDrive/Master/Masterarbeit/Ash/dev/data/journal_list_sociology_test.csv")

    logger.info("END MAIN")
    toc = time.perf_counter()
    logger.info(f"whole script in {toc - tic} seconds")
    '''
    ###
    '''
    fcts.save_data_csv(dtf = counts_df,
                       data_path = data_path,
                       output_file_name = "journal_list_sociology_test")
    '''
    counts_df = fcts.journal_list_wrapper_test(target_addresses = target_addresses,
                                     cores = cores)


if __name__ == "__main__":
    main()


