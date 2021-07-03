'''
This script is recognizing the language of the abstracts

Input:
- Data set including abstracts, titles and labels
'''

import random

random.seed(10)

import configparser

config = configparser.ConfigParser()

import os

config.read(os.getcwd()+'/code/config.ini')

data_path=config['PATH']['data_path']
code_path=config['PATH']['code_path']
project_path=config['PATH']['project']

import pandas as pd

'''
INPUT PARAMETERS HERE
'''
############################################
file_name = "sample_for_damian.csv"
data_orig = pd.read_csv(data_path + "/" + file_name)
sample_size = 10000 #len(data_orig) #10000
field = "abstract" #"title"
save = True #False
##########################################

sample_fraction = sample_size/len(data_orig)
field_unidecode = field + "_unidecode"

data_short = data_orig.sample(frac = sample_fraction).copy()

if (save == True) & (sample_size != len(data_orig)):
    print("we're right here")
    data_short.to_csv(data_path + '/sample_for_damian_' + str(sample_size) + '.csv', index=False)

import en_core_web_sm
from langdetect import detector_factory
from unidecode import unidecode
from tqdm import tqdm
tqdm.pandas()
import time
from spacy_langdetect import LanguageDetector
from spacy.language import Language

def create_lang_detector(nlp, name):
    return LanguageDetector()

Language.factory("language_detector", func=create_lang_detector)

nlp = en_core_web_sm.load()
nlp.add_pipe('language_detector', last=True)


tic = time.perf_counter()
data_short.loc[:,'abstract_unidecode'] = data_short.loc[:,'abstract'].progress_apply(lambda x: unidecode(str(x)))
data_short.loc[:,'title_unidecode'] = data_short.loc[:,'title'].progress_apply(lambda x: unidecode(str(x)))
toc = time.perf_counter()
print(f"unidecode for {len(data_short)} in {toc-tic} seconds")


tic = time.perf_counter()
data_notempty = data_short.loc[:,["Unnamed: 0", field_unidecode]][data_short.loc[:,field_unidecode].str.contains("///")==False]
data_notempty = data_notempty.loc[data_notempty[field_unidecode].str.len()>50,:]
data_notempty.loc[:,"language"] = data_notempty.loc[:,field_unidecode].progress_apply(lambda x: detector_factory.detect_langs(x))
data_notempty.loc[:,"language_short"] = [text[1:3] for text in [str(text) for text in data_notempty.loc[:,"language"]]]
data_notempty.loc[data_notempty["language"].str.len()>1, "language_short"] = "multiple"
toc = time.perf_counter()
print(f"detection for {len(data_notempty)} in {toc-tic} seconds")


'''
ALTERNATIVE, not faster
tic = time.perf_counter()
data_notempty = data_orig[data_orig.abstract.str.contains("///")==False]
data_notempty = data_notempty[data_notempty.abstract.str.len()>50]
index_nonempty = list(data_notempty["Unnamed: 0"])
list_nonempty = list(data_notempty[field_unidecode])
#list_lang = list(map(lambda x: detector_factory.detect_langs(x), tqdm(list_nonempty)))
list_lang = [detector_factory.detect_langs(element) for element in tqdm(list_nonempty)]
list_lang_short = [text[1:3] for text in list_lang]
data_lang = pd.DataFrame({
                    'index': index_nonempty,
                    'lang': list_lang,
                    'lang_short': list_lang_short
                })
#list_nonempty = data_notempty.loc[:,field_unidecode].progress_apply(lambda x: detector_factory.detect_langs(x))
#data_notempty.loc[:,"language"] = data_notempty.loc[:,"language"].astype(str).str[1:3]
toc = time.perf_counter()
print(f"detection for {len(data_notempty)} in {toc-tic} seconds")
'''


data_final = pd.merge(
                data_short.loc[:,data_short.columns != "index"],
                data_notempty.loc[:,["Unnamed: 0", "language", "language_short"]],
                how = "left",
                left_on = "Unnamed: 0",
                right_on = "Unnamed: 0"
            )


data_final.loc[data_final[field_unidecode].str.contains("///")==True, "language_short"] = "multiple"
data_final.loc[data_final[field_unidecode].str.len()<20, "language_short"] = "none"

if save == True:
    if sample_size == len(data_orig):
        print("sample_size == len(data_orig)")
        data_final.to_csv(data_path + '/' + file_name + "_lang.csv", index=False)

    else:
        data_final.to_csv(data_path + '/sample_for_damian_' + str(sample_size) + '_lang.csv', index=False)

