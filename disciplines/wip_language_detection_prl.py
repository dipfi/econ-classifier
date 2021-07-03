'''
This script will classify the abstracts of research papers from different social science disciplines according to their discipline

Input:
- Data set including abstracts, titles and labels
'''

import configparser

config = configparser.ConfigParser()

import os

config.read(os.getcwd()+'/code/config.ini')

data_path=config['PATH']['data_path']
code_path=config['PATH']['code_path']
project_path=config['PATH']['project']

file_name = "sample_for_damian.csv"


import pandas as pd

data_orig = pd.read_csv(data_path + "/" + file_name)


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


samp_size = 1000 #len(data_orig)
field = "abstract" #"title"
field_unidecode = field + "_unidecode"

data_short = data_orig.loc[:samp_size,:].copy()


tic = time.perf_counter()
data_short.loc[:,'abstract_unidecode'] = data_short.loc[:,'abstract'].progress_apply(lambda x: unidecode(str(x)))
data_short.loc[:,'title_unidecode'] = data_short.loc[:,'title'].progress_apply(lambda x: unidecode(str(x)))
toc = time.perf_counter()
print(f"unidecode for {len(data_short)} in {toc-tic} seconds")


tic = time.perf_counter()
data_notempty = data_short.loc[:,["Unnamed: 0", field_unidecode]][data_short.loc[:,field_unidecode].str.contains("///")==False]
data_notempty = data_notempty.loc[data_notempty[field_unidecode].str.len()>50,:]
data_notempty.loc[:,"language"] = data_notempty.loc[:,field_unidecode].progress_apply(lambda x: detector_factory.detect_langs(x))
data_notempty.loc[data_notempty["language"].str.len()>1, "language_short"] = "multiple"
toc = time.perf_counter()
print(f"detection for {len(data_notempty)} in {toc-tic} seconds")


'''
ALTERNATIVE, not faster
tic = time.perf_counter()
data_notempty = data_orig.iloc[:samp_size][data_orig.abstract.str.contains("///")==False]
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

data_final.iloc[:samp_size].to_csv(data_path + '/sample_for_damian_lang.csv', index=False)
