'''
This script will classify the abstracts of research papers from different social science disciplines according to their discipline

Input:
- Data set including abstracts, titles and labels
'''

import pandas as pd

import os

dname = "C:/Users/Damian/kDrive/Main/Master/Masterarbeit/Ash/Project/disciplines/Exploration"
os.chdir(dname)


print(os.getcwd())

data_path = "../../data/"
data_name = "sample_for_damian.csv"

data = pd.read_csv(data_path + data_name)

import en_core_web_sm
from spacy_langdetect import LanguageDetector
nlp = en_core_web_sm.load()
nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)


import time

tic = time.perf_counter()
samp_size = 100
data.loc[:,"language"] = data.loc[:samp_size,"title"].apply(lambda x: nlp(x)._.language["language"])
toc = time.perf_counter()
print(f"{samp_size} in {toc-tic} seconds")





from multiprocessing import Pool
import pandas as pd
import numpy as np


def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def detect_language(x):
    return(nlp(x)._.language["language"])
if __name__ == '__main__':
    #data_par = parallelize_dataframe(data.loc[:samp_size,"title"], detect_language)



