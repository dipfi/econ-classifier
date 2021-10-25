import pandas as pd
import logging
logger = logging.getLogger(__name__)

## for language detection
import langdetect
import multiprocessing as mp

## for preprocessing
import nltk
import re
import os


def hello_world():
    print("hello_world")




def load_data(data_path,
              input_file_name,
              input_file_size,
              input_file_type,
              sample_size):

    logger.info("Function started: load_data")

    if input_file_size == "all":
        input_file_fullname = input_file_name + "." + input_file_type
    else:
        input_file_fullname = input_file_name + "_" + str(input_file_size) + "." + input_file_type

    if input_file_type == "csv":
        logger.info("input_file_type is CSV")
        dtf = pd.read_csv(data_path + "/" + input_file_fullname)
    else:
        logger.info("input_file_type is not CSV")

    if sample_size != "all":
        sample_fraction = sample_size / len(dtf)
        dtf = dtf.sample(frac=sample_fraction).copy()

    logger.info("Data loaded")
    return dtf
    logger.info("Function ended: load_data")



def save_data_csv(dtf,
                  data_path,
                  output_file_name,
                  sample_size):
    logger.info("Function started: save_data_csv")

    if sample_size == "all":
        new_sample_file_name = data_path + '/' + output_file_name + '.csv'

    else:
        new_sample_file_name = data_path + '/' + output_file_name + "_" + str(sample_size) + '.csv'

    logging.info("SAVING NEW SAMPLE FILE: " + new_sample_file_name)
    dtf.to_csv(new_sample_file_name, index=False)

    logger.info("Function ended: save_data_csv")



def rename_fields(dtf,
                   text_field,
                   label_field):
    logger.info("Function started: rename_fields")

    dtf["Journal"] = dtf["Source Title"]
    dtf["Journal"][dtf["Book Series Title"]=="Advances in Austrian Economics"] = dtf["Book Series Title"]

    dtf.rename(columns = {text_field:"text", label_field:"y"}, inplace=True)

    return dtf

    logger.info("Function ended: rename_fields")







def split_text(dtf,
               min_char):
    logger.info("Function started: split_text")

    dtf = dtf.loc[(dtf["text"].str.len() > min_char) ,:].copy()
    dtf['first_part'] = [str(x)[:min_char] for x in dtf.loc[:,"text"]]
    dtf['last_part'] = [str(x)[-min_char:] for x in dtf.loc[:,"text"]]

    return dtf

    logger.info("Function ended: split_text")



def detect_languages(text_series):
    #logger.info("type of text_series: " + str(text_series))
    language_series = [langdetect.detect(text_series) if text_series.strip() != "" else ""]
    return language_series



def multiprocessing_wrapper(input_data,
                            function,
                            cores):

    logger.info("CPU Cores used: " + str(cores))

    pool = mp.Pool(cores)
    results = pool.map(function, input_data)

    pool.close()
    pool.join()

    return results



def add_language_to_file(dtf,
                         language_series_beginning,
                         language_series_end):

    dtf['language_beginning'] = language_series_beginning
    dtf['language_end'] = language_series_end

    dtf.loc[dtf['language_beginning'] == dtf['language_end'], 'lang'] = dtf['language_beginning']
    dtf.loc[dtf['language_beginning'] != dtf['language_end'], 'lang'] = "unassigned"

    return dtf




def language_detection_wrapper(dtf,
                               min_char,
                               cores,
                               function):
    logger.info("Function started: language_detection_wrapper")

    #import logging
    #logger = logging.getLogger("__main__")

    dtf = split_text(dtf = dtf,
                     min_char = min_char)

    language_series_beginning = multiprocessing_wrapper(input_data = dtf['first_part'],
                                                        function = function,
                                                        cores = cores)

    language_series_end = multiprocessing_wrapper(input_data = dtf['last_part'],
                                                  function = function,
                                                  cores = cores)

    dtf = add_language_to_file(dtf = dtf,
                                 language_series_beginning = language_series_beginning,
                                 language_series_end = language_series_end)

    return dtf

    logger.info("Function ended: language_detection_wrapper")







'''
Preprocess a string.
:parameter
    :param text: string - name of column containing text
    :param lst_stopwords: list - list of stopwords to remove
    :param flg_stemm: bool - whether stemming is to be applied
    :param flg_lemm: bool - whether lemmitisation is to be applied
:return
    cleaned text
'''


def utils_preprocess_text(text,
                          flg_stemm=False,
                          flg_lemm=True,
                          remove_last_n = 50):

    lst_stopwords = nltk.corpus.stopwords.words("english")

    subtitles = ["design", "methodology", "approach", "originality", "value", "limitations", "implications"]
    lst_stopwords.extend(subtitles)

    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', ' ', str(text[:-remove_last_n]).lower().strip())

    ## Tokenize (convert from string to list)
    lst_text = text.split()

    ## remove Stopwords
    lst_text = [word for word in lst_text if word not in lst_stopwords]

    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    ## back to string from list
    text = " ".join(lst_text)
    return text

#dtf["text_clean"] = dtf["text"].apply(lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, lst_stopwords=lst_stopwords))



def preprocessing_wrapper(dtf,
                          cores,
                          function=utils_preprocess_text):
    logger.info("Function started: preprocessing_wrapper")

    #import logging
    #logger = logging.getLogger("__main__")

    #lst_stopwords = nltk.corpus.stopwords.words("english")

    proproc_series = multiprocessing_wrapper(input_data = dtf["text"],
                                                function = function,
                                                cores = cores)

    dtf["text_clean"] = proproc_series

    return dtf

    logger.info("Function ended: preprocessing_wrapper")