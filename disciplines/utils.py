def import_statements():
    ## set up
    import random
    import configparser
    import os
    import time
    from tqdm import tqdm

    ## for data
    import pandas as pd
    import collections
    import json
    from scipy import stats

    ## for plotting
    import matplotlib.pyplot as plt
    import seaborn as sns
    import wordcloud

    ## for text processing
    import re
    import nltk

    ## for language detection
    import langdetect

    ## for sentiment
    from textblob import TextBlob## for ner
    import spacy

    ## for vectorizer
    from sklearn import feature_extraction, manifold

    ## for word embedding
    import gensim.downloader as gensim_api

    ## for topic modeling
    import gensim

def set_config():
    config.read(os.getcwd()+'/code/config.ini')
    data_path=config['PATH']['data_path']
    code_path=config['PATH']['code_path']
    project_path=config['PATH']['project']