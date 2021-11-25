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
import csv

import pickle

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


##parallelization
import multiprocessing as mp



'''
CLASSIFIER SPECIFIC IMPORTS
'''

## for data
import pandas as pd
pd.set_option('display.max_columns', None)


## for data
import json
import numpy as np
from scipy import stats

## for plotting
#import matplotlib as mpl
#mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

## for processing
import re
import nltk

## for language detection
import langdetect

## for bag-of-words
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing, feature_selection, metrics, linear_model, ensemble, svm

## for balancing data
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

## for explainer
from lime import lime_text
import wordcloud

## for word embedding
import gensim
import gensim.downloader as gensim_api


## for deep learning
from tensorflow.keras import models, layers, preprocessing as kprocessing


from tensorflow.keras import backend as K

## for bert language model
import transformers





############################################
logging_level = logging.INFO  # logging.DEBUG #logging.WARNING
print_charts_tables = True  # False #True
input_file_name = "WOS_lee_heterodox_und_samequality_new_preprocessed_2"
text_field_clean = "text_clean"  # "title" #"abstract"
text_field = "text"
label_field = "y"
cores = mp.cpu_count()  #mp.cpu_count()  #2
plot = 0 #0 = none, 1 = some, 2 = all

save_results = True
results_file_name = "4Journals_BERT"

use_model = False
save_model = False
model_file_name = "3All_Data_BERT" #"WOS_lee_heterodox_und_samequality_new_preprocessed_tfidf_model" #"WOS_lee_heterodox_und_samequality_new_preprocessed_w2v_model" #"WOS_lee_heterodox_und_samequality_new_preprocessed_bert_model"

save_weights = False

use_training_samples=False
save_training_samples=False

train_on_all = False
test_size = 0.1 #suggestion: 0.1
training_set = "oversample" # "oversample", "undersample", "heterodox", "samequality" ; suggestion: oversample

use_reproducible_train_test_split = False
train_set_name = "WOS_lee_heterodox_und_samequality_preprocessed_train_9"
test_set_name = "WOS_lee_heterodox_und_samequality_preprocessed_test_1"

journal_split = True
num_journals = "all" #3 #"all"
random_journals = False
journal_list = [i for i in range(70,75)] #False # [65,1]


#TFIDF only
tfidf = False
min_df_list = [3, 5, 10] #[1000, 5000, 10000]
p_value_limit_list = [0.0, 0.5, 0.75, 0.9] #[0.8, 0.9, 0.95]
ngram_range_list = [(1,1), (1,3)] #[(1,1), (1,2), (1,3)]
tfidf_classifier_list = ["LogisticRegression"] #["naive_bayes", "LogisticRegression", "RandomForestClassifier","GradientBoostingClassifier", "SVC"]

#w2v only
w2v = False
use_gigaword = False #if True the pretrained model "glove-wiki-gigaword-[embedding_vector_length]d" is used
use_embeddings = False #if True a trained model needs to be selected below
which_embeddings = False #"word2vec_numabs_909_embedlen_300_embedepo_10_window_8_embed_False" #specify model to use here
embedding_folder = "embeddings"
train_new = True #if True new embeddings are trained

num_epochs_for_embedding_list = [5] #number of epochs to train the word embeddings ; sugegstion: 10 (embedding_set = "False")
num_epochs_for_classification_list= [15] #number of epochs to train the the classifier ; suggetion: 10 (with 300 dim. embeddings)
embedding_vector_length_list = [300] #suggesion: 300

window_size_list = [12] #suggesion: 8

embedding_only = False
embedding_set = False # "oversample", "undersample", "heterodox", "samequality", False ; suggestion: False

max_length_of_document_vector_w2v_list = [100] #np.max([len(i.split()) for i in X_train_series]) #np.quantile([len(i.split()) for i in X_train_series], 0.7) ; suggesion: 100
classifier_loss_function_w2v_list = ['sparse_categorical_crossentropy'] #, 'mean_squared_error', 'sparse_categorical_crossentropy', "kl_divergence", 'categorical_hinge'
w2v_batch_size_list = [256] #suggestion: 256

#BERT only
bert = True
small_model_list = [True]
bert_batch_size_list = [64] #suggestion 64
bert_epochs_list = [12]
max_length_of_document_vector_bert_list = [200] #np.max([len(i.split()) for i in X_train_series]) #np.quantile([len(i.split()) for i in X_train_series], 0.7) ; suggesion: 350
classifier_loss_function_bert_list = ['sparse_categorical_crossentropy'] #, 'mean_squared_error', 'sparse_categorical_crossentropy', "kl_divergence", 'categorical_hinge'
use_bert_feature_matrix = False
save_bert_feature_matrix = False

############################################

parameters = """PARAMETERS:
input_file_name = """ + input_file_name + """
cores = """ + str(cores) + """
save_results = """ + str(save_results) + """
journal_split = """ + str(journal_split) + """
num_journals = """ + str(num_journals) + """
random_journals = """ + str(random_journals) + """
journal_list = """ + str(journal_list) + """
test_size = """ + str(test_size) + """
training_set = """ + str(training_set) + """
use_reproducible_train_test_split = """ + str(use_reproducible_train_test_split) + """
train_set_name = """ + str(train_set_name) + """
test_set_name = """ + str(test_set_name) + """
tfidf = """ + str(tfidf) + """
w2v = """ + str(w2v) + """
bert = """ + str(bert)

if tfidf:
    parameters_tfidf = """PARAMETERS TFIDF:
    min_df_list = """ + str(min_df_list) + """
    p_value_limit_list = """ + str(p_value_limit_list) + """
    ngram_range_list = """ + str(ngram_range_list) + """
    tfidf_classifier_list = """ + str(tfidf_classifier_list)



if w2v:
    parameters_w2v = """PARAMETERS W2V:
    use_gigaword = """ + str(use_gigaword) + """
    use_embeddings = """ + str(use_embeddings) + """
    embedding_folder = """ + str(embedding_folder) + """
    train_new = """ + str(train_new) + """
    num_epochs_for_embedding_list = """ + str(num_epochs_for_embedding_list) + """
    embedding_vector_length_list = """ + str(embedding_vector_length_list) + """
    num_epochs_for_classification_list = """ + str(num_epochs_for_classification_list) + """
    window_size_list = """ + str(window_size_list) + """
    embedding_only = """ + str(embedding_only) + """
    embedding_set = """ + str(embedding_set) + """
    max_length_of_document_vector_w2v_list = """ + str(max_length_of_document_vector_w2v_list) + """
    classifier_loss_function_w2v_list = """ + str(classifier_loss_function_w2v_list) + """
    w2v_batch_size_List = """ + str(w2v_batch_size_list)

if bert:
    parameters_bert = """PARAMETERS BERT:
    max_length_of_document_vector_bert_list = """ + str(max_length_of_document_vector_bert_list) + """
    classifier_loss_function_bert_list = """ + str(classifier_loss_function_bert_list) + """
    small_model_list = """ + str(small_model_list) + """
    bert_batch_size_list = """ + str(bert_batch_size_list) + """
    bert_epochs_list = """ + str(bert_epochs_list) + """
    use_bert_feature_matrix = """ + str(use_bert_feature_matrix) + """
    save_bert_feature_matrix = """ + str(save_bert_feature_matrix)


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



if results_file_name == False:

    if tfidf:
        results_file_name = input_file_name + "_tfidf_results"
    if w2v:
        results_file_name = input_file_name + "_w2v_results"
    if bert:
        results_file_name = input_file_name + "_bert_results"
        
if model_file_name == False:

    if tfidf:
        model_file_name = input_file_name + "_tfidf_model"
    if w2v:
        model_file_name = input_file_name + "_w2v_model"
    if bert:
        model_file_name = input_file_name + "_bert_model"




if use_model:

    '''
    LOAD DATA
    '''
    logger.info("LOAD DATA")
    logger.info("LOAD DATA")

    dtf = pd.read_csv(data_path + "/" + input_file_name + ".csv")

    dtf["index"] = np.arange(dtf.shape[0])



    X_test = dtf[text_field_clean].values

    dic_y_mapping = {0: "0orthodox",
                     1: "1heterodox"}
    inverse_dic = {v: k for k, v in dic_y_mapping.items()}

    if tfidf:
        model_file_path = (data_path + "/models/" + model_file_name + ".pkl")

        with open(model_file_path, 'rb') as file:
            model_loaded = pickle.load(file)

        logger.info("TEST CALC")
        predicted = model_loaded.predict(X_test)
        predicted_prob = model_loaded.predict_proba(X_test)
        predicted_bin = np.array([np.argmax(pred) for pred in predicted_prob])
        logger.info("TEST CALC FINISHED")

        if save_weights:
            weights_loaded_path = data_path + "/weights/" + model_file_name + "_tfidf_weights.csv"

            weights_loaded_dtf = pd.read_csv(weights_loaded_path)
            weights_loaded_dict = {k: v for k, v in zip(weights_loaded_dtf["words"], weights_loaded_dtf["weights"])}

            corpus = dtf[text_field_clean]
            dtf_weights = pd.DataFrame()
            for prediction in range(2):
                idx = [idx for idx, i in enumerate(predicted_bin) if i == prediction]
                corpus_loop = corpus[idx]
                ## create list of n-grams
                lst_corpus = []
                for string in corpus_loop:
                    lst_words = string.split()
                    lst_grams = [" ".join(lst_words[i:i + 1]) for i in range(0, len(lst_words), 1)]
                    lst_corpus.append(lst_grams)

                dtf_corpus = pd.DataFrame({"words": [i for l in lst_corpus for i in l]})
                dtf_corpus = dtf_corpus["words"].value_counts()

                def catch(dict, w):
                    try:
                        return dict[w]
                    except:
                        return None

                weights_list = [catch(weights_loaded_dict, w) for w in dtf_corpus.index.tolist()]
                dtf_weights = dtf_weights.append(pd.DataFrame({"words": dtf_corpus.index.tolist(),
                                                            "counts": dtf_corpus.values.tolist(),
                                                            "weights": weights_list,
                                                            "prediction": [prediction for i in weights_list]})).copy()

            dtf_weights["weighted_counts"] = dtf_weights["counts"]*dtf_weights["weights"]
            dtf_weights.to_csv(data_path + "/weights/input_" + input_file_name + "_model_" + model_file_name + "_tfidf_weights.csv", index=False)







    if w2v:
        model_file_path = (data_path + "/models/" + model_file_name)

        model_loaded = models.load_model(model_file_path)

        corpus = dtf[text_field_clean]


        ## create list of n-grams
        lst_corpus = []
        for string in corpus:
            lst_words = string.split()
            lst_grams = [" ".join(lst_words[i:i + 1]) for i in range(0, len(lst_words), 1)]
            lst_corpus.append(lst_grams)


        """
        tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ', oov_token="NaN", filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(lst_corpus)
        dic_vocabulary = tokenizer.word_index
        """

        # loading tokenizer

        tokenizer_file_path = (data_path + "/models/" + model_file_name + "_tokenizer.pkl")
        with open(tokenizer_file_path, 'rb') as file:
            tokenizer = pickle.load(file)


        ## detect bigrams and trigrams
        bigrams_detector = gensim.models.phrases.Phrases(lst_corpus, delimiter=' ', min_count=5, threshold=10)
        bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
        trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[lst_corpus], delimiter=" ", min_count=2, threshold=10)
        trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)

        ## detect common bigrams and trigrams using the fitted detectors
        lst_corpus = list(bigrams_detector[lst_corpus])
        lst_corpus = list(trigrams_detector[lst_corpus])

        ## text to sequence with the fitted tokenizer
        lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)

        ## padding sequence
        max_length_of_document_vector_w2v = max_length_of_document_vector_w2v_list[0]
        X_test = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=max_length_of_document_vector_w2v, padding="post", truncating="post")

        """
        #load embeddings
        if which_embeddings == False:

            if journal_split:
                modelname = "word2vec_" + str(test_journal.replace(" ", "_")) + "_numabs_" + str(len(dtf)) + "_embedlen_" + str(embedding_vector_length) + "_embedepo_" + str(num_epochs_for_embedding) + "_window_" + str(window_size) + "_embed_" + str(embedding_set)
            else:
                modelname = "word2vec_numabs_" + str(len(dtf)) + "_embedlen_" + str(embedding_vector_length) + "_embedepo_" + str(num_epochs_for_embedding) + "_window_" + str(window_size) + "_embed_" + str(embedding_set)

            pretrained_vectors = str(data_path) + "/" + embedding_folder + "/" + modelname
        else:
            pretrained_vectors = str(data_path) + "/" + embedding_folder + "/" + which_embeddings

        nlp = gensim.models.word2vec.Word2Vec.load(pretrained_vectors)


        ## start the matrix (length of vocabulary x vector size) with all 0s
        embedding_vector_length = embedding_vector_length_list[0]
        embeddings = np.zeros((len(dic_vocabulary) + 1, embedding_vector_length))

        if use_gigaword:
            for word, idx in dic_vocabulary.items():
                ## update the row with vector
                try:
                    embeddings[idx] = nlp[word]
                ## if word not in model then skip and the row stays all 0s
                except:
                    pass

        else:
            for word, idx in dic_vocabulary.items():
                ## update the row with vector
                try:
                    embeddings[idx] = nlp.wv[word]
                ## if word not in model then skip and the row stays all 0s
                except:
                    pass
        """

        predicted_prob = model_loaded.predict(X_test)
        predicted = [dic_y_mapping[np.argmax(pred)] for pred in predicted_prob]
        predicted_bin = np.array([np.argmax(pred) for pred in predicted_prob])

        """     
        layer = [layer for layer in model_loaded.layers if "attention" in layer.name][0]
        func = K.function([model_loaded.input], [layer.output])

        weights = np.zeros((len(X_test), max_length_of_document_vector_w2v))
        for i in range(0, len(X_test), 2500):
            weight = func(X_test[i:i + 2500])[0]
            weight = np.mean(weight, axis=2)
            weights[i:i + 2500] = weight

        # weights = np.mean(weights, axis=2).flatten()

        ### 3. rescale weights, remove null vector, map word-weight
        weights = [preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(np.array(i).reshape(-1, 1)).reshape(-1) for i in weights]

        inv_word_dic = {v: k for k, v in tokenizer.word_index.items()}

        dtf_weights_long = pd.DataFrame()
        for prediction in range(2):
            idx = [idx for idx, i in enumerate(predicted_bin) if i == prediction]
            weights_loop = [weights[i] for i in idx]
            weights_loop = [i for s in weights for i in s]
            words = [X_test[i] for i in idx]
            words = [i for s in words for i in s]
            weights_loop = [weights_loop[n] for n, idx in enumerate(words) if idx > 1]
            words = [inv_word_dic[i] for i in words if i > 1]
            dtf_weights_long = dtf_weights_long.append(pd.DataFrame({"words": words, "weights": weights_loop, "prediction": [prediction for i in words]})).copy()
        dtf_weights = dtf_weights_long.groupby(["words", "prediction"]).mean().copy()
        dtf_weights["count"] = dtf_weights_long.groupby(["words", "prediction"]).count()["weights"].to_list()

        dtf_weights.to_csv(data_path + "/weights/input_" + input_file_name + "_model_" + model_file_name + "_w2v_weights.csv")
        """



    if bert:

        model_file_path = (data_path + "/models/" + model_file_name)

        model_loaded = models.load_model(model_file_path)

        small_model = small_model_list[0]

        if small_model:
            try:
                tokenizer = transformers.AutoTokenizer.from_pretrained(data_path + '/distilbert-base-uncased/', do_lower_case=True)
            except:
                tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        else:
            try:
                tokenizer = transformers.AutoTokenizer.from_pretrained(data_path + '/bert-base-uncased/', do_lower_case=True)
            except:
                tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        # Feature engineer Test set
        logger.info("Feature engineer Test set")

        X_test_ids = dtf["index"].tolist()
        max_length_of_document_vector_bert = max_length_of_document_vector_bert_list[0]
        maxqnans = np.int((max_length_of_document_vector_bert - 5))# / 2)

        text_lst = [text[:-50] for text in dtf[text_field]]
        text_lst = [' '.join(text.split()[:maxqnans]) for text in text_lst]
        # text_lst = [text for text in text_lst if text]

        corpus = text_lst

        ## add special tokens
        logger.info("add special tokens test")
        corpus_tokenized = ["[CLS] " +
                            " ".join(tokenizer.tokenize(re.sub(r'[^\w\s]+|\n', '', str(txt).lower().strip()))[:maxqnans]) +
                            " [SEP] " for txt in corpus]

        ## generate masks
        logger.info("generate masks test")
        masks = [[1] * len(txt.split(" ")) + [0] * (max_length_of_document_vector_bert - len(txt.split(" "))) for txt in corpus_tokenized]

        ## padding
        logger.info("padding test")
        txt2seq = [txt + " [PAD]" * (max_length_of_document_vector_bert - len(txt.split(" "))) if len(txt.split(" ")) != max_length_of_document_vector_bert else txt for txt in corpus_tokenized]

        ## generate idx
        logger.info("generate idx test")

        if use_bert_feature_matrix:
            idx_frozen = pd.read_csv(data_path + "/" + str(input_file_name) + "_" + str(max_length_of_document_vector_bert) + "_bert_feature_matrix.csv",
                                     delimiter=",", header=None).values.tolist()

            idx = [idx_frozen[i] for i in X_test_ids]

        else:
            idx = [tokenizer.encode(seq.split(" "), is_split_into_words=True) for seq in txt2seq]
            minlen = min([len(i) for i in idx])
            idx = [i[:max_length_of_document_vector_bert] for i in idx]

            if save_bert_feature_matrix:
                np.savetxt(data_path + "/" + str(input_file_name) + "_" + str(max_length_of_document_vector_bert) + "_bert_feature_matrix.csv",
                           idx,
                           delimiter=",",
                           fmt='% s')

        ## feature matrix
        logger.info("feature matrix test")

        if small_model:
            #X_test = [np.array(idx, dtype='int32')]
            X_test = [np.array(idx, dtype='int32'), np.array(masks, dtype='int32')]

        else:
            ## generate segments
            logger.info("generate segments")
            segments = []
            for seq in txt2seq:
                temp, i = [], 0
                for token in seq.split(" "):
                    temp.append(i)
                    if token == "[SEP]":
                        i += 1
                segments.append(temp)

            X_test = [np.array(idx, dtype='int32'), np.array(masks, dtype='int32'), np.array(segments, dtype='int32')]



        predicted_prob = model_loaded.predict(X_test)
        predicted = [dic_y_mapping[np.argmax(pred)] for pred in predicted_prob]
        predicted_bin = np.array([np.argmax(pred) for pred in predicted_prob])

    now = time.asctime()

    try:
        journal = dtf["Journal"].str.lower().tolist()
    except:
        journal = [None for i in range(len(dtf))]
    try:
        pubyear = dtf["Publication Year"].tolist()
    except:
        pubyear = [None for i in range(len(dtf))]
    try:
        WOS_ID = dtf["UT (Unique WOS ID)"].tolist()
    except:
        WOS_ID = [None for i in range(len(dtf))]
    try:
        author = dtf["Author Full Names"].tolist()
    except:
        author = [None for i in range(len(dtf))]
    try:
        title = dtf["Article Title"].tolist()
    except:
        title = [None for i in range(len(dtf))]
    try:
        abstract = dtf["Abstract"].tolist()
    except:
        abstract = [None for i in range(len(dtf))]
    try:
        times_cited = dtf["Times Cited, All Databases"].tolist()
    except:
        times_cited = [None for i in range(len(dtf))]
    try:
        abstract = dtf[text_field].tolist()
    except:
        try:
            abstract = dtf["Abstract"].tolist()
        except:
            abstract = [None for i in range(len(dtf))]
    try:
        WOS_category = dtf["WoS Categories"].tolist()
    except:
        WOS_category = [None for i in range(len(dtf))]
    try:
        research_area = dtf["Research Areas"].tolist()
    except:
        research_area = [None for i in range(len(dtf))]
    try:
        label = dtf[label_field].tolist()
    except:
        try:
            label = dtf["labels"].tolist()
        except:
            label = [None for i in range(len(dtf))]
    try:
        keywords_author = dtf["Author Keywords"].tolist()
    except:
        keywords_author = [None for i in range(len(dtf))]
    try:
        keywords_plus = dtf["Keywords Plus"].tolist()
    except:
        keywords_plus = [None for i in range(len(dtf))]

    results_df = pd.DataFrame({"time": [now for i in predicted],
                               "input_data": [input_file_name for i in predicted],
                               "model_file_name": [model_file_name for i in predicted],
                               "journal": journal,
                               "pubyear": pubyear,
                               "author": author,
                               "times_cited": times_cited,
                               "predicted": predicted,
                               "predicted_bin": predicted_bin,
                               "predicted_prob": predicted_prob[:, 1],
                               "label": label,
                               "title": title,
                               "abstract": abstract,
                               "research_area": research_area,
                               "WOS_category": WOS_category,
                               "WOS_ID": WOS_ID,
                               "keywords_author": keywords_author,
                               "keywords_plus": keywords_plus})

    results_path = data_path + "/results/" + results_file_name + ".csv"

    if save_results:
        results_df.to_csv(results_path, index=False)






else:
    logger.info(parameters)

    if tfidf:
        logger.info(parameters_tfidf)

    if w2v:
        logger.info(parameters_w2v)

    if bert:
        logger.info(parameters_bert)

    from Utils import utils_ortho_hetero as fcts

    '''
    LOAD DATA
    '''
    logger.info("LOAD DATA")
    logger.info("LOAD DATA")

    dtf = pd.read_csv(data_path + "/" + input_file_name + ".csv")
    dtf["index"] = np.arange(dtf.shape[0])

    if plot == 1 or plot == 2:
        fig, ax = plt.subplots()
        fig.suptitle("Label Distribution in Original Data", fontsize=12)
        dtf[label_field].reset_index().groupby(label_field).count().sort_values(by="index").plot(kind="barh", legend=False, ax=ax).grid(axis='x')
        plt.show()












#######################################################################3

    '''
    TRAIN TEST SPLIT
    '''
    logger.info("TRAIN TEST SPLIT")

    dtf["index"] = np.arange(dtf.shape[0])

    if journal_split:

        all_journals = dtf[["Journal", label_field]].drop_duplicates().copy()
        all_journals["Journal"] = all_journals["Journal"].str.lower().copy()
        all_journals = all_journals.drop_duplicates().copy()

        if random_journals:
            all_journals = all_journals.sample(frac = 1).copy()

        if journal_list != False:
            try:
                all_journals = all_journals.iloc[journal_list]
            except:
                all_journals = all_journals[all_journals["Journal"].isin(journal_list)]

    else:
        all_journals = pd.DataFrame({"Journal": ["randmom"], label_field: ["random"]})
        test_journal = None
        test_label = None



    if num_journals == "all":
        last_journal = len(all_journals)
    else:
        last_journal = num_journals




    loop_number = 0

    for index, all_test in all_journals.iterrows():
        loop_number = loop_number + 1

        if loop_number > last_journal:
            break

        logger.info("Loop Nr. = " + str(loop_number))

        logger.info("TRAIN TEST SPLIT")

        if journal_split == True:

            test_label = all_test[label_field]

            test_journal = all_test["Journal"]

            logger.info("Journal = " + str(test_journal))
            logger.info("Label = " + str(test_label))

            dtf_train = dtf.loc[dtf["Journal"].str.lower() != test_journal].copy()
            dtf_test = dtf.loc[dtf["Journal"].str.lower() == test_journal].copy()

            training_set_id = ''.join(test_journal.split()) + str(int(time.time()*1000))

        else:
            if use_reproducible_train_test_split:
                dtf_train = pd.read_csv(data_path + "/" + train_set_name + ".csv")
                dtf_test = pd.read_csv(data_path + "/" + test_set_name + ".csv")
                training_set_id = "use_reproducible_train_test_split"
                dtf_train["index"] = np.arange(dtf_train.shape[0])
                dtf_test["index"] = np.arange(dtf_test.shape[0])

            else:
                dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size = test_size, random_state=42)
                training_set_id = "random" + str(int(time.time()*1000))


        if train_on_all:
            dtf_train = dtf.copy()
            dtf_test = dtf_train.copy()

        #balanbce dataset
        logger.info("BALANCE TRAINING SET")


        if training_set == "oversample":
            over_sampler = RandomOverSampler(random_state=42)
            X_train, y_train = over_sampler.fit_resample(pd.DataFrame({"X": dtf_train[text_field_clean], "X_raw": dtf_train[text_field], "ID": dtf_train["index"]}), pd.DataFrame({"y": dtf_train[label_field]}))
            X_train_ids = X_train["ID"].tolist()
            X_train_raw = pd.DataFrame({"X": X_train["X_raw"].tolist()})
            X_train = pd.DataFrame({"X": X_train["X"].tolist()})

        elif training_set == "undersample":
            under_sampler = RandomUnderSampler(random_state=42)
            X_train, y_train = under_sampler.fit_resample(pd.DataFrame({"X": dtf_train[text_field_clean], "X_raw": dtf_train[text_field], "ID": dtf_train["index"]}), pd.DataFrame({"y": dtf_train[label_field]}))
            X_train_ids = X_train["ID"].tolist()
            X_train_raw = pd.DataFrame({"X": X_train["X_raw"].tolist()})
            X_train = pd.DataFrame({"X": X_train["X"].tolist()})

        else:
            X_train = pd.DataFrame({"X": dtf_train[text_field_clean], "X_raw": dtf_train[text_field], "ID": dtf_train["index"]})
            X_train_ids = X_train["ID"].tolist()
            y_train = pd.DataFrame({"y": dtf_train[label_field]})
            X_train_raw = pd.DataFrame({"X": X_train["X_raw"].tolist()})
            X_train = pd.DataFrame({"X": X_train["X"].tolist()})


        training_samples_file_path = (data_path + "/models/" + input_file_name + "_training_samples.csv")

        if use_training_samples:
            training_samples = pd.read_csv(training_samples_file_path)

            X_train_ids = training_samples["X_train_ids"].tolist()
            y_train = pd.DataFrame({"y": training_samples["y_train"].tolist()})
            X_train_raw = pd.DataFrame({"X": training_samples["X_train_raw"].tolist()})
            X_train = pd.DataFrame({"X": training_samples["X_train"].tolist()})

        if save_training_samples and save_model and not use_training_samples:
            training_samples = pd.DataFrame({"X_train": X_train["X"],
                                             "X_train_raw": X_train_raw["X"],
                                             "X_train_ids": X_train_ids,
                                             "y_train": y_train["y"]})

            training_samples.to_csv(training_samples_file_path, index=False)

        X_train_series = X_train.squeeze(axis=1)
        X_train_raw_series = X_train_raw.squeeze(axis=1)
        y_train_series = y_train.squeeze(axis=1)




        if plot == 1 or plot == 2:
            fig, ax = plt.subplots()
            fig.suptitle("Label Distribution in Training Data", fontsize=12)
            y_train_series.value_counts().plot(kind="barh", legend=False, ax=ax).grid(axis='x')
            plt.show()

        if plot == 1 or plot == 2:
            fig, ax = plt.subplots()
            fig.suptitle("Label Distribution in Embedding Data", fontsize=12)
            y_embed_series.value_counts().plot(kind="barh", legend=False, ax=ax).grid(axis='x')
            plt.show()


        trainingset_id = "TRAININGSET" + str(int(time.time() * 1000))

        models_list = []

        if tfidf == True:
            models_list += ["tfidf"]

        if w2v == True:
            models_list += ["w2v"]

        if bert == True:
            models_list += ["bert"]







        for current_model in models_list:





##########################################################################

            if current_model == "tfidf":

                #TFIDF
                logger.info("TFIDF")

                loop_tfidf_classifier = 0

                for tfidf_classifier in tfidf_classifier_list:

                    loop_tfidf_classifier += 1

                    if tfidf_classifier == "naive_bayes":
                        classifier = naive_bayes.MultinomialNB()
                    elif tfidf_classifier == "LogisticRegression":
                        classifier = linear_model.LogisticRegression()
                    elif tfidf_classifier == "LogisticRegressionCV":
                        classifier = linear_model.LogisticRegressionCV(max_iter=300)
                    elif tfidf_classifier == "SVC":
                        classifier = svm.SVC(probability=True)
                    elif tfidf_classifier == "RandomForestClassifier":
                        classifier = ensemble.RandomForestClassifier()
                    elif tfidf_classifier == "GradientBoostingClassifier":
                        classifier = ensemble.GradientBoostingClassifier()

                    loop_min_df = 0
                    for min_df in min_df_list:

                        loop_min_df += 1
                        loop_ngram_range = 0

                        for ngram_range in ngram_range_list:

                            loop_ngram_range += 1
                            loop_p_value_limit = 0

                            for p_value_limit in p_value_limit_list:
                                loop_p_value_limit += 1

                                logger.info("Loop tfidf_classifier Nr.: " + str(loop_tfidf_classifier))
                                logger.info("tfidf_classifier: " + str(tfidf_classifier))
                                logger.info("Loop min_df Nr: " + str(loop_min_df))
                                logger.info("Loop min_df: " + str(min_df))
                                logger.info("Loop ngram_range Nr: " + str(loop_ngram_range))
                                logger.info("Loop ngram_range: " + str(ngram_range))
                                logger.info("Loop p_value_limit Nr: " + str(loop_p_value_limit))
                                logger.info("Loop p_value_limit: " + str(p_value_limit))

                                class_time_start = time.perf_counter()

                                vectorizer = feature_extraction.text.TfidfVectorizer(min_df=min_df, ngram_range=ngram_range)

                                X_train_unbalanced = X_train[0:len(dtf_train)]
                                y_train_unbalanced = y_train[0:len(dtf_train)]

                                corpus_unbalanced = X_train_unbalanced["X"]

                                vectorizer.fit(corpus_unbalanced)
                                X_train_new_unbalanced = vectorizer.transform(corpus_unbalanced)
                                dic_vocabulary = vectorizer.vocabulary_

                                if plot == 2:
                                    sns.heatmap(X_train_new_unbalanced.todense()[:, np.random.randint(0, X_train_new_unbalanced.shape[1], 100)] == 0, vmin=0, vmax=1, cbar=False).set_title('Sparse Matrix Sample')

                                # FEATURE SELECTION
                                logger.info("FEATURE SELECTION")
                                y = y_train_unbalanced
                                X_names_unbalanced = vectorizer.get_feature_names()




                                dtf_features = pd.DataFrame()
                                for cat in np.unique(y):
                                    logger.info("cat: " + str(cat))
                                    chi2, p = feature_selection.chi2(X_train_new_unbalanced, y == cat)
                                    dtf_features = dtf_features.append(pd.DataFrame({"feature": X_names_unbalanced, "score": 1 - p, "y": cat}))
                                    dtf_features = dtf_features.sort_values([label_field, "score"], ascending=[True, False])
                                    dtf_features = dtf_features[dtf_features["score"] > p_value_limit]

                                X_names_new_unbalanced = dtf_features["feature"].unique().tolist()

                                # shorter
                                logger.info("SHORTENING VOCABULARY")

                                if len(X_names_new_unbalanced) > 0:
                                    vectorizer_new = feature_extraction.text.TfidfVectorizer(vocabulary=X_names_new_unbalanced)
                                else:
                                    vectorizer_new = feature_extraction.text.TfidfVectorizer(vocabulary=X_names_unbalanced)
                                    p_value_limit = "no limit"



                                corpus = X_train["X"]
                                vectorizer_new.fit(corpus)
                                X_train_new2 = vectorizer_new.transform(corpus)
                                dic_vocabulary = vectorizer_new.vocabulary_

                                # classify
                                logger.info("SET UP CLASSIFIER")

                                ## pipeline
                                model = pipeline.Pipeline([("vectorizer_new", vectorizer_new),
                                                           ("classifier", classifier)])

                                ## train classifier
                                logger.info("TRAIN CLASSIFIER")

                                y_train_new = y_train.values.ravel()

                                model["classifier"].fit(X_train_new2, y_train_new)

                                if save_model:
                                    model_file_path = (data_path + "/models/" + model_file_name + ".pkl")
                                    with open(model_file_path, 'wb') as file:
                                        pickle.dump(model, file)

                                    '''
                                    joblib.dump(model["classifier"], )
                                    '''
                                ## encode y
                                logger.info("encode y")
                                dic_y_mapping = {n: label for n, label in enumerate(np.unique(y_train_new))}
                                inverse_dic = {v: k for k, v in dic_y_mapping.items()}
                                y_train_bin = np.array([inverse_dic[y] for y in y_train_new])


                                ## test
                                logger.info("TEST CLASSIFIER")

                                X_test = dtf_test[text_field_clean].values
                                y_test = dtf_test[label_field].values.ravel()

                                predicted = model.predict(X_test)
                                predicted_prob = model.predict_proba(X_test)

                                logger.info("TEST CALC")
                                predicted_bin = np.array([np.argmax(pred) for pred in predicted_prob])
                                logger.info("TEST CALC FINISHED")

                                y_test_bin = np.array([inverse_dic[y] for y in y_test])
                                classes = np.array([dic_y_mapping[0], dic_y_mapping[1]])

                                class_time_total = time.perf_counter() - class_time_start
                                logger.info(f"classification with {min_df} features and ngram_range {ngram_range} for {len(dtf)} samples in {class_time_total} seconds")

                                # results allg
                                now = time.asctime()
                                result_id = "RESULT" + str(int(time.time() * 1000))

                                result_all = pd.DataFrame({"time": [now],
                                                       "trainingset_id": [trainingset_id],
                                                       "result_id": [result_id],
                                                       "length_data": [len(dtf)],
                                                       "length_training_orig": [len(dtf_train)],
                                                       "length_training_samp": [len(X_train_raw_series)],
                                                       "test_journal": [test_journal],
                                                       "test_label": [test_label],
                                                       "use_reproducible_train_test_split": [use_reproducible_train_test_split],
                                                       "train_set_name": [train_set_name],
                                                       "test_set_name": [test_set_name],
                                                       "tfidf": [tfidf],
                                                       "w2v": [w2v],
                                                       "bert": [bert],
                                                       "duration": [class_time_total],
                                                       "current_model": [current_model]})

                                # results tfidf
                                result_tfidf = pd.DataFrame({"min_df": [min_df],
                                                             "p_value_limit": [p_value_limit],
                                                             "ngram_range": [ngram_range],
                                                             "tfidf_classifier": [tfidf_classifier],
                                                             "number_relevant_features": [len(X_names_new_unbalanced)]})

                                ## test
                                y_test = dtf_test[label_field].values


                                ### EVALUATION
                                result_fct = fcts.evaluate(classes=classes,
                                                           # y_test = y_test,
                                                           y_test_bin=y_test_bin,
                                                           # predicted = predicted,
                                                           predicted_bin=predicted_bin,
                                                           predicted_prob=predicted_prob)

                                result = pd.concat([result_all, result_fct, result_tfidf], axis=1)

                                logger.info("RESULT DETAILS:")
                                logger.info(result)

                                if save_results:
                                    logger.info("SAVING RESULTS")

                                    results, pred_prob_df = fcts.save_results(data_path=data_path,
                                                                              results_file_name=results_file_name,
                                                                              result=result,
                                                                              dtf_test=dtf_test,
                                                                              trainingset_id=trainingset_id,
                                                                              result_id=result_id,
                                                                              predicted_prob=predicted_prob,
                                                                              y_test_bin=y_test_bin)

                                if save_weights:
                                    words = vectorizer_new.get_feature_names()
                                    weights = [i for i in model["classifier"].coef_[0]]
                                    dtf_weights = pd.DataFrame({"words": words, "weights" : weights})

                                    weights_loaded_dict = {k: v for k, v in zip(dtf_weights["words"], dtf_weights["weights"])}

                                    corpus = dtf_test[text_field_clean]
                                    dtf_weights = pd.DataFrame()
                                    for prediction in range(2):
                                        idx = [idx for idx, i in enumerate(predicted_bin) if i == prediction]
                                        corpus_loop = corpus.iloc[idx]
                                        ## create list of n-grams
                                        lst_corpus = []
                                        for string in corpus_loop:
                                            lst_words = string.split()
                                            lst_grams = [" ".join(lst_words[i:i + 1]) for i in range(0, len(lst_words), 1)]
                                            lst_corpus.append(lst_grams)

                                        dtf_corpus = pd.DataFrame({"words": [i for l in lst_corpus for i in l]})
                                        dtf_corpus = dtf_corpus["words"].value_counts()


                                        def catch(dict, w):
                                            try:
                                                return dict[w]
                                            except:
                                                return None


                                        weights_list = [catch(weights_loaded_dict, w) for w in dtf_corpus.index.tolist()]
                                        dtf_weights = dtf_weights.append(pd.DataFrame({"words": dtf_corpus.index.tolist(),
                                                                                       "prediction": [prediction for i in weights_list],
                                                                                       "weights": weights_list,
                                                                                       "counts": dtf_corpus.values.tolist()})).copy()

                                    dtf_weights["weighted_counts"] = dtf_weights["counts"] * dtf_weights["weights"]
                                    dtf_weights.to_csv(data_path + "/weights/" + model_file_name + "_tfidf_weights.csv", index=False)










###################################################################################################
            ###################################################################################################
            ###################################################################################################







            if current_model == "w2v":
                gigaword_loaded = False

                if embedding_set == "oversample":
                    over_sampler = RandomOverSampler(random_state=42)
                    X_embed, y_embed = over_sampler.fit_resample(pd.DataFrame({"X": dtf_train[text_field_clean]}), pd.DataFrame({"y": dtf_train[label_field]}))

                elif embedding_set == "undersample":
                    under_sampler = RandomUnderSampler(random_state=42)
                    X_embed, y_embed = under_sampler.fit_resample(pd.DataFrame({"X": dtf_train[text_field_clean]}), pd.DataFrame({"y": dtf_train[label_field]}))

                elif embedding_set == "heterodox":
                    X_embed = pd.DataFrame({"X": dtf_train.loc[dtf_train[label_field] == "heterodox"][text_field_clean]})
                    y_embed = pd.DataFrame({"y": dtf_train.loc[dtf_train[label_field] == "heterodox"][label_field]})

                elif embedding_set == "samequality":
                    X_embed = pd.DataFrame({"X": dtf_train.loc[dtf_train[label_field] == "samequality"][text_field_clean]})
                    y_embed = pd.DataFrame({"y": dtf_train.loc[dtf_train[label_field] == "samequality"][label_field]})

                else:
                    X_embed = pd.DataFrame({"X": dtf_train[text_field_clean]})
                    y_embed = pd.DataFrame({"y": dtf_train[label_field]})

                X_embed_series = X_embed.squeeze(axis=1)
                y_embed_series = y_embed.squeeze(axis=1)

                loop_num_epochs_for_embedding = 0

                for num_epochs_for_embedding in num_epochs_for_embedding_list:

                    loop_num_epochs_for_embedding += 1
                    loop_embedding_vector_length = 0

                    for embedding_vector_length in embedding_vector_length_list:

                        loop_embedding_vector_length += 1
                        loop_window_size = 0

                        for window_size in window_size_list:

                            loop_window_size += 1

                            logger.info("Loop num_epochs_for_embedding Nr.: " + str(loop_num_epochs_for_embedding))
                            logger.info("num_epochs_for_embedding: " + str(num_epochs_for_embedding))
                            logger.info("Loop embedding_vector_length Nr.: " + str(loop_num_epochs_for_embedding))
                            logger.info("embedding_vector_length: " + str(embedding_vector_length))
                            logger.info("Loop window_size Nr.: " + str(loop_window_size))
                            logger.info("window_size: " + str(window_size))

                            # FEATURE ENGINEERING

                            logger.info("FEATURE ENGINEERING")

                            logger.info("FEATURE ENGINEERING FOR TRAINING SET")

                            corpus = X_embed_series

                            ## create list of lists of unigrams
                            lst_corpus = []
                            for string in corpus:
                                lst_words = string.split()
                                lst_grams = [' '.join(lst_words[i:i + 1]) for i in range(0, len(lst_words), 1)]
                                lst_corpus.append(lst_grams)

                            ## detect bigrams and trigrams
                            bigrams_detector = gensim.models.phrases.Phrases(lst_corpus, delimiter=' ', min_count=5, threshold=10)
                            bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
                            trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[lst_corpus], delimiter=" ", min_count=2, threshold=10)
                            trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)

                            logger.info("STARTING WORD EMBEDDING")

                            ## fit w2v

                            if use_gigaword:
                                if gigaword_loaded != True:
                                    gigaword_start = time.perf_counter()

                                    modelname_raw = "glove-wiki-gigaword-" + str(embedding_vector_length)

                                    if journal_split:
                                        modelname = modelname_raw + "_word2vec_" + str(test_journal.replace(" ", "_")) + "_numabs_" + str(len(dtf))
                                    else:
                                        modelname = modelname_raw + "_numabs_" + str(len(dtf))

                                    pretrained_vectors = modelname_raw

                                    nlp = gensim_api.load(pretrained_vectors)

                                    word = "bad"
                                    nlp[word].shape
                                    nlp.most_similar(word)

                                    ## word embedding
                                    tot_words = [word] + [tupla[0] for tupla in nlp.most_similar(word, topn=20)]
                                    X = nlp[tot_words]

                                    gigaword_loaded = True

                                    gigaword_end = time.perf_counter()
                                    gigaword_time = gigaword_end - gigaword_start
                                    logger.info(f"loading gigaword vectors in {gigaword_time} seconds")

                            if use_embeddings:
                                load_embeddings_start = time.perf_counter()

                                if journal_split:
                                    modelname = "word2vec_" + str(test_journal.replace(" ", "_")) + "_numabs_" + str(len(dtf)) + "_embedlen_" + str(embedding_vector_length) + "_embedepo_" + str(num_epochs_for_embedding) + "_window_" + str(window_size) + "_embed_" + str(embedding_set)
                                else:
                                    modelname = "word2vec_numabs_" + str(len(dtf)) + "_embedlen_" + str(embedding_vector_length) + "_embedepo_" + str(num_epochs_for_embedding) + "_window_" + str(window_size) + "_embed_" + str(embedding_set)

                                if which_embeddings == False:
                                    pretrained_vectors = str(data_path) + "/" + embedding_folder + "/" + modelname
                                else:
                                    pretrained_vectors = str(data_path) + "/" + embedding_folder + "/" + which_embeddings

                                nlp = gensim.models.word2vec.Word2Vec.load(pretrained_vectors)



                                '''
                                word = "bad"
                                logger.info(str(nlp.wv[word].shape))
                                logger.info(word + ": " + str(nlp.wv.most_similar(word)))

                                ## word embedding
                                tot_words = [word] + [tupla[0] for tupla in nlp.wv.most_similar(word, topn=20)]
                                X = nlp.wv[tot_words]

                                load_embeddings_end = time.perf_counter()
                                load_embeddings_time = load_embeddings_end - load_embeddings_start
                                logger.info(f"loading embeddings in {load_embeddings_time} seconds")
                                '''

                            if train_new:
                                train_embeddings_start = time.perf_counter()
                                if journal_split:
                                    modelname_raw = "word2vec_" + str(test_journal.replace(" ", "_")) + "_numabs_" + str(len(dtf)) + "_embedlen_" + str(embedding_vector_length) + "_embedepo_" + str(num_epochs_for_embedding) + "_window_" + str(window_size) + "_embed_" + str(embedding_set)

                                else:
                                    modelname_raw = "word2vec_numabs_" + str(len(dtf)) + "_embedlen_" + str(embedding_vector_length) + "_embedepo_" + str(num_epochs_for_embedding) + "_window_" + str(window_size) + "_embed_" + str(embedding_set)

                                modelname = "newembedding_" + str(modelname_raw)

                                nlp = gensim.models.word2vec.Word2Vec(lst_corpus, vector_size=embedding_vector_length, window=window_size, sg=1, epochs=num_epochs_for_embedding, workers=cores)

                                nlp.save(str(data_path) + "/" + embedding_folder + "/" + modelname_raw)

                                train_embeddings_end = time.perf_counter()
                                train_embeddings_time = train_embeddings_end - train_embeddings_start
                                logger.info(f"training word2vec for {len(dtf)} documents and {num_epochs_for_embedding} epochs in {train_embeddings_time} seconds")

                            logger.info("WORD EMBEDDING FINISHED")

                            if embedding_only == False:

                                logger.info("START TOKENIZE")
                                ## tokenize text
                                tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ', oov_token="NaN", filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
                                tokenizer.fit_on_texts(lst_corpus)
                                dic_vocabulary = tokenizer.word_index

                                tokenizer_file_path = (data_path + "/models/" + model_file_name + "_tokenizer.pkl")
                                with open(tokenizer_file_path, 'wb') as file:
                                    pickle.dump(tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)

                                logger.info("START CREATING CORPUS")

                                corpus = X_train_series

                                ## create list of n-grams
                                lst_corpus = []
                                for string in corpus:
                                    lst_words = string.split()
                                    lst_grams = [" ".join(lst_words[i:i + 1]) for i in range(0, len(lst_words), 1)]
                                    lst_corpus.append(lst_grams)

                                ## detect common bigrams and trigrams using the fitted detectors
                                lst_corpus = list(bigrams_detector[lst_corpus])
                                lst_corpus = list(trigrams_detector[lst_corpus])

                                ## create sequence
                                lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)

                                loop_max_length_of_document_vector_w2v = 0

                                for max_length_of_document_vector_w2v in max_length_of_document_vector_w2v_list:

                                    loop_max_length_of_document_vector_w2v += 1
                                    logger.info("Loop max_length_of_document_vector_w2v Nr: " + str(loop_max_length_of_document_vector_w2v))
                                    logger.info("max_length_of_document_vector_w2v: " + str(max_length_of_document_vector_w2v))

                                    class_time_start = time.perf_counter()

                                    ## padding sequence

                                    X_train_new = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=max_length_of_document_vector_w2v, padding="post", truncating="post")


                                    logger.info("SET UP EMBEDDING MATRIX")

                                    ## start the matrix (length of vocabulary x vector size) with all 0s
                                    embeddings = np.zeros((len(dic_vocabulary) + 1, embedding_vector_length))

                                    if use_gigaword:
                                        for word, idx in dic_vocabulary.items():
                                            ## update the row with vector
                                            try:
                                                embeddings[idx] = nlp[word]
                                            ## if word not in model then skip and the row stays all 0s
                                            except:
                                                pass

                                    else:
                                        for word, idx in dic_vocabulary.items():
                                            ## update the row with vector
                                            try:
                                                embeddings[idx] = nlp.wv[word]
                                            ## if word not in model then skip and the row stays all 0s
                                            except:
                                                pass

                                    loop_classifier_loss_function_w2v = 0

                                    for classifier_loss_function_w2v in classifier_loss_function_w2v_list:
                                        loop_classifier_loss_function_w2v += 1
                                        logger.info("Loop classifier_loss_function_w2v Nr: " + str(loop_classifier_loss_function_w2v))
                                        logger.info("classifier_loss_function_w2v: " + str(classifier_loss_function_w2v))

                                        logger.info("NETWORK ARCHITECTURE")


                                        ## code attention layer
                                        def attention_layer(inputs, neurons):
                                            x = layers.Permute((2, 1))(inputs)
                                            x = layers.Dense(neurons, activation="softmax")(x)
                                            x = layers.Permute((2, 1), name="attention")(x)
                                            x = layers.multiply([inputs, x])
                                            return x


                                        ## input
                                        x_in = layers.Input(shape=(max_length_of_document_vector_w2v,))

                                        ## embedding
                                        x = layers.Embedding(input_dim=embeddings.shape[0],
                                                             output_dim=embeddings.shape[1],
                                                             weights=[embeddings],
                                                             input_length=max_length_of_document_vector_w2v, trainable=False)(x_in)

                                        ## apply attention
                                        x = attention_layer(x, neurons=max_length_of_document_vector_w2v)

                                        ## 2 layers of bidirectional lstm
                                        x = layers.Bidirectional(layers.LSTM(units=max_length_of_document_vector_w2v, dropout=0.2, return_sequences=True))(x)
                                        x = layers.Bidirectional(layers.LSTM(units=max_length_of_document_vector_w2v, dropout=0.2))(x)

                                        ## final dense layers
                                        x = layers.Dense(64, activation='relu')(x)
                                        y_out = layers.Dense(2, activation='softmax')(x)

                                        ## compile
                                        model = models.Model(x_in, y_out)
                                        model.compile(loss=classifier_loss_function_w2v, optimizer='adam', metrics=['accuracy'])

                                        if print_charts_tables:
                                            model.summary()

                                        logger.info("ENCODING FEATURES")

                                        ## encode y
                                        dic_y_mapping = {n: label for n, label in enumerate(np.unique(y_train))}
                                        inverse_dic = {v: k for k, v in dic_y_mapping.items()}
                                        y_train_bin = np.array([inverse_dic[y] for y in y_train['y']])

                                        loop_num_epochs_for_classification = 0
                                        for num_epochs_for_classification in num_epochs_for_classification_list:

                                            loop_num_epochs_for_classification += 1
                                            loop_w2v_batch_size = 0

                                            for w2v_batch_size in w2v_batch_size_list:

                                                loop_w2v_batch_size += 1

                                                logger.info("Loop num_epochs_for_classification Nr: " + str(loop_num_epochs_for_classification))
                                                logger.info("num_epochs_for_classification: " + str(num_epochs_for_classification))
                                                logger.info("Loop w2v_batch_size Nr: " + str(loop_w2v_batch_size))
                                                logger.info("w2v_batch_size: " + str(w2v_batch_size))

                                                logger.info("STARTING TRAINING")

                                                ## train
                                                train_start = time.perf_counter()
                                                model.fit(x=X_train_new, y=y_train_bin, batch_size=w2v_batch_size, epochs=num_epochs_for_classification, shuffle=True, verbose=0, validation_split=0.3, workers=cores)

                                                if save_model:
                                                    model_file_path = (data_path + "/models/" + model_file_name)
                                                    model.save(model_file_path)

                                                class_time_total = time.perf_counter() - class_time_start
                                                logger.info(f"classification with {num_epochs_for_classification} epochs batch size {w2v_batch_size} for {len(dtf)} samples in {class_time_total} seconds")

                                                # results allg
                                                now = time.asctime()
                                                result_id = "RESULT" + str(int(time.time() * 1000))

                                                result_all = pd.DataFrame({"time": [now],
                                                                           "trainingset_id": [trainingset_id],
                                                                           "result_id": [result_id],
                                                                           "length_data": [len(dtf)],
                                                                           "length_training_orig": [len(dtf_train)],
                                                                           "length_training_samp": [len(X_train_raw_series)],
                                                                           "test_journal": [test_journal],
                                                                           "test_label": [test_label],
                                                                           "use_reproducible_train_test_split": [use_reproducible_train_test_split],
                                                                           "train_set_name": [train_set_name],
                                                                           "test_set_name": [test_set_name],
                                                                           "tfidf": [tfidf],
                                                                           "w2v": [w2v],
                                                                           "bert": [bert],
                                                                           "duration": [class_time_total],
                                                                           "current_model": [current_model]})

                                                # results w2v
                                                result_w2v = pd.DataFrame({"use_gigaword": [use_gigaword],
                                                                              "use_embeddings": [use_embeddings],
                                                                              "embedding_folder": [embedding_folder],
                                                                              "train_new": [train_new],
                                                                              "embedding_vector_length": [embedding_vector_length],
                                                                              "num_epochs_for_embedding": [num_epochs_for_embedding],
                                                                              "window_size": [window_size],
                                                                              "embedding_only": [embedding_only],
                                                                              "embedding_set": [embedding_set],
                                                                              "max_length_of_document_vector_w2v": [max_length_of_document_vector_w2v],
                                                                              "classifier_loss_function_w2v": [classifier_loss_function_w2v],
                                                                              "w2v_batch_size": [w2v_batch_size],
                                                                           "num_epochs_for_classification": [num_epochs_for_classification]})



                                                logger.info("FEATURE ENGINEERING FOR TEST SET")

                                                corpus = dtf_test[text_field_clean]

                                                ## create list of n-grams
                                                lst_corpus = []
                                                for string in corpus:
                                                    lst_words = string.split()
                                                    lst_grams = [" ".join(lst_words[i:i + 1]) for i in range(0, len(lst_words), 1)]
                                                    lst_corpus.append(lst_grams)

                                                ## detect common bigrams and trigrams using the fitted detectors
                                                lst_corpus = list(bigrams_detector[lst_corpus])
                                                lst_corpus = list(trigrams_detector[lst_corpus])

                                                ## text to sequence with the fitted tokenizer
                                                lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)

                                                ## padding sequence
                                                X_test = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=max_length_of_document_vector_w2v, padding="post", truncating="post")

                                                ## test
                                                predicted_prob = model.predict(X_test)
                                                predicted = [dic_y_mapping[np.argmax(pred)] for pred in predicted_prob]
                                                y_test = dtf_test[label_field].values
                                                predicted_bin = np.array([np.argmax(pred) for pred in predicted_prob])
                                                y_test_bin = np.array([inverse_dic[y] for y in y_test])
                                                classes = np.array([dic_y_mapping[0], dic_y_mapping[1]])

                                                ### EVALUATION
                                                result_fct = fcts.evaluate(classes=classes,
                                                                           # y_test = y_test,
                                                                           y_test_bin=y_test_bin,
                                                                           # predicted = predicted,
                                                                           predicted_bin=predicted_bin,
                                                                           predicted_prob=predicted_prob)

                                                result = pd.concat([result_all, result_fct, result_w2v], axis=1)

                                                logger.info("RESULT DETAILS:")
                                                logger.info(result)

                                                if save_results:
                                                    logger.info("SAVING RESULTS")

                                                    results, pred_prob_df = fcts.save_results(data_path=data_path,
                                                                                              results_file_name=results_file_name,
                                                                                              result=result,
                                                                                              dtf_test=dtf_test,
                                                                                              trainingset_id=trainingset_id,
                                                                                              result_id=result_id,
                                                                                              predicted_prob=predicted_prob,
                                                                                              y_test_bin=y_test_bin)

                                                """
                                                predicted_prob = model.predict(X_train_new)
                                                predicted_bin_train = np.array([np.argmax(pred) for pred in predicted_prob])

                                                
                                                layer = [layer for layer in model.layers if "attention" in layer.name][0]
                                                func = K.function([model.input], [layer.output])

                                                weights = np.zeros((len(X_train_new), max_length_of_document_vector_w2v))
                                                for i in range(0, len(X_train_new), 2500):
                                                    weight = func(X_train_new[i:i+2500])[0]
                                                    weight = np.mean(weight, axis=2)
                                                    weights[i:i+2500] = weight



                                                #weights = np.mean(weights, axis=2).flatten()

                                                ### 3. rescale weights, remove null vector, map word-weight
                                                weights = [preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(np.array(i).reshape(-1, 1)).reshape(-1) for i in weights]

                                                inv_word_dic = {v: k for k, v in tokenizer.word_index.items()}

                                                dtf_weights = pd.DataFrame()
                                                for prediction in range(2):
                                                    idx = [idx for idx, i in enumerate(predicted_bin) if i == prediction]
                                                    weights_loop = [weights[i] for i in idx]
                                                    weights_loop = [i for s in weights for i in s]
                                                    words = [X_train_new[i] for i in idx]
                                                    words = [i for s in words for i in s]
                                                    weights_loop = [weights_loop[n] for n, idx in enumerate(words) if idx != 0]
                                                    words = [inv_word_dic[i] for i in words if i != 0]
                                                    dtf_weights = dtf_weights.append(pd.DataFrame({"words": words, "weights": weights_loop, "prediction": [prediction for i in words]})).copy()
                                                dtf_weights = dtf_weights.groupby(["words","prediction"]).mean().copy()

                                                dtf_weights.to_csv(data_path + "/weights/" + model_file_name + "_w2v_weights.csv")
                                                """



                                                """
                                                layer = [layer for layer in model.layers if "attention" in layer.name][0]
                                                func = K.function([model.input], [layer.output])

                                                weights = np.zeros((len(X_test), max_length_of_document_vector_w2v))
                                                for i in range(0, len(X_test), 2500):
                                                    weight = func(X_test[i:i + 2500])[0]
                                                    weight = np.mean(weight, axis=2)
                                                    weights[i:i + 2500] = weight

                                                # weights = np.mean(weights, axis=2).flatten()

                                                ### 3. rescale weights, remove null vector, map word-weight
                                                weights = [preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(np.array(i).reshape(-1, 1)).reshape(-1) for i in weights]

                                                inv_word_dic = {v: k for k, v in tokenizer.word_index.items()}

                                                dtf_weights_long = pd.DataFrame()
                                                for prediction in range(2):
                                                    idx = [idx for idx, i in enumerate(predicted_bin) if i == prediction]
                                                    weights_loop = [weights[i] for i in idx]
                                                    weights_loop = [i for s in weights for i in s]
                                                    words = [X_test[i] for i in idx]
                                                    words = [i for s in words for i in s]
                                                    weights_loop = [weights_loop[n] for n, idx in enumerate(words) if idx > 1]
                                                    words = [inv_word_dic[i] for i in words if i > 1]
                                                    dtf_weights_long = dtf_weights_long.append(pd.DataFrame({"words": words, "weights": weights_loop, "prediction": [prediction for i in words]})).copy()
                                                dtf_weights = dtf_weights_long.groupby(["words", "prediction"]).mean().copy()
                                                dtf_weights["count"] = dtf_weights_long.groupby(["words", "prediction"]).count()["weights"].to_list()


                                                dtf_weights.to_csv(data_path + "/weights/" + model_file_name + "_w2v_weights.csv")
                                                """











            if current_model == "bert":

                small_model_loop = 0

                for small_model in small_model_list:

                    small_model_loop += 1

                    if small_model:
                        tokenizer = transformers.AutoTokenizer.from_pretrained(data_path + '/distilbert-base-uncased/', do_lower_case=True)

                        """
                        try:
                            tokenizer = transformers.AutoTokenizer.from_pretrained(data_path + '/distilbert-base-uncased/', do_lower_case=True)
                        except:
                            tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
                    else:
                        try:
                            tokenizer = transformers.AutoTokenizer.from_pretrained(data_path + '/bert-base-uncased/', do_lower_case=True)
                        except:
                            tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
                        """
                    max_length_of_document_vector_bert_loop = 0

                    for max_length_of_document_vector_bert in max_length_of_document_vector_bert_list:

                        max_length_of_document_vector_bert_loop += 1
                        logger.info("Loop small_model Nr: " + str(small_model_loop))
                        logger.info("small_model : " + str(small_model))
                        logger.info("Loop max_length_of_document_vector_bert Nr: " + str(max_length_of_document_vector_bert_loop))
                        logger.info("max_length_of_document_vector_bert : " + str(max_length_of_document_vector_bert))

                        text_lst = [text[:-50] for text in X_train_raw["X"]]
                        text_lst = [' '.join(text.split()[:max_length_of_document_vector_bert]) for text in text_lst]

                        subtitles = ["design", "methodology", "approach", "originality", "value", "limitations", "implications", "elsevier", "purpose"]

                        text_lst = [word for word in text_lst if word not in subtitles]

                        # text_lst = [text for text in text_lst if text]

                        corpus = text_lst

                        ## Fearute engineering train set
                        logger.info("Fearute engineering train set")

                        ## add special tokens
                        logger.info("add special tokens")

                        maxqnans = np.int((max_length_of_document_vector_bert - 5)) # / 2)
                        corpus_tokenized = ["[CLS] " +
                                            " ".join(tokenizer.tokenize(re.sub(r'[^\w\s]+|\n', '',str(txt).lower().strip()))[:maxqnans]) +
                                            " [SEP] " for txt in corpus]

                        ## generate masks
                        logger.info("generate masks")
                        masks = [[1] * len(txt.split(" ")) + [0] * (max_length_of_document_vector_bert - len(txt.split(" "))) for txt in corpus_tokenized]

                        ## padding
                        logger.info("padding")
                        txt2seq = [txt + " [PAD]" * (max_length_of_document_vector_bert - len(txt.split(" "))) if len(txt.split(" ")) != max_length_of_document_vector_bert else txt for txt in corpus_tokenized]

                        ## generate idx
                        logger.info("generate idx")


                        if use_bert_feature_matrix:
                            idx_frozen = pd.read_csv(data_path + "/" + str(input_file_name) + "_" + str(max_length_of_document_vector_bert) + "_bert_feature_matrix.csv",
                                              delimiter=",", header = None).values.tolist()

                            idx = [idx_frozen[i-1] for i in X_train_ids]

                        else:
                            idx = [tokenizer.encode(seq.split(" "), is_split_into_words=True) for seq in txt2seq]
                            minlen = min([len(i) for i in idx])
                            idx = [i[:max_length_of_document_vector_bert] for i in idx]

                            if save_bert_feature_matrix:
                                np.savetxt(data_path + "/" + str(train_set_name) + "_" + str(max_length_of_document_vector_bert) + "_bert_feature_matrix.csv",
                                           idx,
                                           delimiter=",",
                                           fmt='% s')

                        ## feature matrix
                        logger.info("feature matrix")

                        if small_model:
                            X_train_new = [np.asarray(idx, dtype='int32'), np.asarray(masks, dtype='int32')]

                        else:

                            ## generate segments
                            logger.info("generate segments")
                            segments = []
                            for seq in txt2seq:
                                temp, i = [], 0
                                for token in seq.split(" "):
                                    temp.append(i)
                                    if token == "[SEP]":
                                        i += 1
                                segments.append(temp)

                            X_train_new = [np.asarray(idx, dtype='int32'), np.asarray(masks, dtype='int32'), np.asarray(segments, dtype='int32')]





                        ##CLASSIFIER
                        logger.info("CLASSIFIER")

                        classifier_loss_function_bert_loop = 0

                        for classifier_loss_function_bert in classifier_loss_function_bert_list:
                            classifier_loss_function_bert_loop += 1
                            logger.info("classifier_loss_function_bert_loop Nr = " + str(classifier_loss_function_bert_loop))
                            logger.info("classifier_loss_function_bert = " + str(classifier_loss_function_bert))

                            if small_model:

                                ##DISTIL-BERT
                                logger.info("DISTIL-BERT")

                                ## inputs
                                idx = layers.Input((max_length_of_document_vector_bert), dtype="int32", name="input_idx")
                                masks = layers.Input((max_length_of_document_vector_bert), dtype="int32", name="input_masks")
                                # segments = layers.Input((max_length_of_document_vector_bert), dtype="int32", name="input_segments")

                                ## pre-trained bert with config
                                config = transformers.DistilBertConfig(dropout=0.2, attention_dropout=0.2)
                                config.output_hidden_states = False

                                try:
                                    nlp = transformers.TFDistilBertModel.from_pretrained(data_path + '/distilbert-base-uncased/', config=config)
                                except:
                                    nlp = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased', config=config)

                                bert_out = nlp.distilbert(idx, attention_mask=masks)[0]

                                ## fine-tuning
                                x = layers.GlobalAveragePooling1D()(bert_out)
                                x = layers.Dense(64, activation="relu")(x)
                                y_out = layers.Dense(len(np.unique(y_train)), activation='softmax')(x)

                                ## compile
                                logger.info("compile DISTIL-BERT")
                                model = models.Model([idx, masks], y_out)

                                for layer in model.layers[:3]:
                                    layer.trainable = False

                                model.compile(loss=classifier_loss_function_bert, optimizer='adam', metrics=['mse'])

                                model.summary()

                            else:
                                ##BERT
                                logger.info("BERT")

                                ## inputs
                                idx = layers.Input((max_length_of_document_vector_bert), dtype="int32", name="input_idx")
                                masks = layers.Input((max_length_of_document_vector_bert), dtype="int32", name="input_masks")
                                segments = layers.Input((max_length_of_document_vector_bert), dtype="int32", name="input_segments")

                                ## pre-trained bert
                                try:
                                    nlp = transformers.TFBertModel.from_pretrained(data_path + '/bert-base-uncased/')
                                except:
                                    nlp = transformers.TFBertModel.from_pretrained('bert-base-uncased')

                                bert_out = nlp.bert([idx, masks, segments])
                                '''
                                sequence_out = bert_out.last_hidden_state
                                pooled_out = bert_out.pooler_output
                                '''
                                seq_out, _ = bert_out[0], bert_out[1]
                                ## fine-tuning
                                x = layers.GlobalAveragePooling1D() (seq_out)
                                x = layers.Dense(64, activation="relu")(x)
                                y_out = layers.Dense(len(np.unique(y_train)), activation='softmax')(x)

                                ## compile
                                logger.info("compile BERT")
                                model = models.Model([idx, masks, segments], y_out)

                                for layer in model.layers[:4]:
                                    layer.trainable = False

                                model.compile(loss = classifier_loss_function_bert, optimizer='adam', metrics=['mse'])

                                model.summary()







                            # Feature engineer Test set
                            logger.info("Feature engineer Test set")

                            X_test_ids = dtf_test["index"].tolist()

                            text_lst = [text[:-50] for text in dtf_test[text_field]]
                            text_lst = [' '.join(text.split()[:maxqnans]) for text in text_lst]
                            #text_lst = [text for text in text_lst if text]

                            corpus = text_lst

                            ## add special tokens
                            logger.info("add special tokens test")
                            maxqnans = np.int((max_length_of_document_vector_bert - 5))# / 2)
                            corpus_tokenized = ["[CLS] " +
                                                " ".join(tokenizer.tokenize(re.sub(r'[^\w\s]+|\n', '',str(txt).lower().strip()))[:maxqnans]) +
                                                " [SEP] " for txt in corpus]

                            ## generate masks
                            logger.info("generate masks test")
                            masks = [[1] * len(txt.split(" ")) + [0] * (max_length_of_document_vector_bert - len(txt.split(" "))) for txt in corpus_tokenized]

                            ## padding
                            logger.info("padding test")
                            txt2seq = [txt + " [PAD]" * (max_length_of_document_vector_bert - len(txt.split(" "))) if len(txt.split(" ")) != max_length_of_document_vector_bert else txt for txt in corpus_tokenized]

                            ## generate idx
                            logger.info("generate idx test")

                            if use_bert_feature_matrix:
                                idx_frozen = pd.read_csv(data_path + "/" + str(input_file_name) + "_" + str(max_length_of_document_vector_bert) + "_bert_feature_matrix.csv",
                                                  delimiter=",", header = None).values.tolist()

                                idx = [idx_frozen[i] for i in X_test_ids]

                            else:
                                idx = [tokenizer.encode(seq.split(" "), is_split_into_words=True) for seq in txt2seq]
                                minlen = min([len(i) for i in idx])
                                idx = [i[:max_length_of_document_vector_bert] for i in idx]

                                if save_bert_feature_matrix:
                                    np.savetxt(data_path + "/" + str(input_file_name) + "_" + str(max_length_of_document_vector_bert) + "_bert_feature_matrix.csv",
                                               idx,
                                               delimiter=",",
                                               fmt='% s')


                            ## feature matrix
                            logger.info("feature matrix test")

                            if small_model:
                                X_test = [np.array(idx, dtype='int32'), np.array(masks, dtype='int32')]

                            else:
                                ## generate segments
                                logger.info("generate segments")
                                segments = []
                                for seq in txt2seq:
                                    temp, i = [], 0
                                    for token in seq.split(" "):
                                        temp.append(i)
                                        if token == "[SEP]":
                                            i += 1
                                    segments.append(temp)

                                X_test = [np.array(idx, dtype='int32'), np.array(masks, dtype='int32'), np.array(segments, dtype='int32')]





                            ## encode y
                            logger.info("encode y")
                            dic_y_mapping = {n:label for n,label in enumerate(np.unique(y_train))}
                            inverse_dic = {v:k for k,v in dic_y_mapping.items()}
                            y_train_bin = np.array([inverse_dic[y] for y in y_train["y"]])

                            bert_batch_size_loop = 0

                            for bert_batch_size in bert_batch_size_list:

                                bert_batch_size_loop += 1

                                bert_epochs_loop = 0

                                for bert_epochs in bert_epochs_list:

                                    class_time_start = time.perf_counter()

                                    bert_epochs_loop += 1

                                    logger.info("max_length_of_document_vector_bert_loop Nr: " + str(max_length_of_document_vector_bert_loop))
                                    logger.info("max_length_of_document_vector_bert : " + str(max_length_of_document_vector_bert))
                                    logger.info("bert_batch_size Nr: " + str(bert_batch_size_loop))
                                    logger.info("bert_batch_size : " + str(bert_batch_size))
                                    logger.info("bert_epochs_loop Nr: " + str(bert_epochs_loop))
                                    logger.info("bert_epochs : " + str(bert_epochs))

                                    ## train
                                    model.fit(x=X_train_new, y=y_train_bin, batch_size=bert_batch_size, epochs=bert_epochs, shuffle=True, verbose=1, validation_split=0.3)

                                    if save_model:
                                        model_file_path = str(data_path + "/models/" + model_file_name)
                                        saving = model.save(model_file_path)

                                    class_time_total = time.perf_counter() - class_time_start
                                    logger.info(f"classification with {bert_epochs} epochs batch size {bert_batch_size} for {len(dtf)} samples in {class_time_total} seconds")



                                    #results allg
                                    now = time.asctime()
                                    result_id = "RESULT" + str(int(time.time() * 1000))

                                    result_all = pd.DataFrame({"time": [now],
                                                               "trainingset_id": [trainingset_id],
                                                               "result_id": [result_id],
                                                               "length_data": [len(dtf)],
                                                               "length_training_orig": [len(dtf_train)],
                                                               "length_training_samp": [len(X_train_raw_series)],
                                                               "test_journal": [test_journal],
                                                               "test_label": [test_label],
                                                               "use_reproducible_train_test_split": [use_reproducible_train_test_split],
                                                               "train_set_name": [train_set_name],
                                                               "test_set_name": [test_set_name],
                                                               "tfidf": [tfidf],
                                                               "w2v": [w2v],
                                                               "bert": [bert],
                                                               "duration": [class_time_total],
                                                               "current_model": [current_model]})


                                    # results bert
                                    result_bert = pd.DataFrame({"max_length_of_document_vector_bert": [max_length_of_document_vector_bert],
                                                              "classifier_loss_function_bert": [classifier_loss_function_bert],
                                                              "small_model": [small_model],
                                                              "bert_batch_size": [bert_batch_size],
                                                              "bert_epochs": [bert_epochs]})


                                    ## test
                                    predicted_prob = model.predict(X_test)
                                    predicted = [dic_y_mapping[np.argmax(pred)] for pred in predicted_prob]
                                    y_test = dtf_test[label_field].values
                                    predicted_bin = np.array([np.argmax(pred) for pred in predicted_prob])
                                    y_test_bin = np.array([inverse_dic[y] for y in y_test])
                                    classes = np.array([dic_y_mapping[0], dic_y_mapping[1]])




                                    ### EVALUATION
                                    result_fct = fcts.evaluate(classes = classes,
                                                                 #y_test = y_test,
                                                                 y_test_bin = y_test_bin,
                                                                 #predicted = predicted,
                                                                 predicted_bin = predicted_bin,
                                                                 predicted_prob = predicted_prob)

                                    result = pd.concat([result_all, result_fct, result_bert], axis = 1)

                                    logger.info("RESULT DETAILS:")
                                    logger.info(result)

                                    if save_results:
                                        logger.info("SAVING RESULTS")

                                        results, pred_prob_df = fcts.save_results(data_path=data_path,
                                                                                 results_file_name=results_file_name,
                                                                                 result=result,
                                                                                 dtf_test=dtf_test,
                                                                                 trainingset_id=trainingset_id,
                                                                                 result_id=result_id,
                                                                                 predicted_prob=predicted_prob,
                                                                                 y_test_bin=y_test_bin)








toc = time.perf_counter()
logger.info(f"whole script for {len(dtf)} in {toc-tic} seconds")
print("the end")
