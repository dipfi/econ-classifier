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
