def import():
    ## set up
    import random
    import time
    import os
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
