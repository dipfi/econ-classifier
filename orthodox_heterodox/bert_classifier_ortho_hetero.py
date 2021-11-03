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
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing, feature_selection, metrics

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
input_file_name = "WOS_lee_heterodox_und_samequality_preprocessed_1000"
input_file_size = "all" #10000 #"all"
input_file_type = "csv"
output_file_name = "WOS_lee_heterodox_und_samequality_preprocessed_wip"
sample_size = "all"  #input_file_size #10000 #"all"
text_field_clean = "text_clean"  # "title" #"abstract"
text_field = "text"
label_field = "y"
cores = mp.cpu_count()  #mp.cpu_count()  #2
save = False  # False #True
plot = 0 #0 = none, 1 = some, 2 = all
use_gigaword = False #if True the pretrained model "glove-wiki-gigaword-[embedding_vector_length]d" is used
use_embeddings = False #if True a trained model needs to be selected below
#which_embeddings = "word2vec_numabs_79431_embedlen_300_epochs_30" #specify model to use here
embedding_folder = "embeddings"
train_new = True #if True new embeddings are trained

num_epochs_for_embedding_list = [12, 15, 18] #number of epochs to train the word embeddings ; sugegstion: 10(-15) (undersampling training)
num_epochs_for_classification_list= [15] #number of epochs to train the the classifier ; suggetion: 10 (with 300 dim. embeddings)
embedding_vector_length_list = [300] #suggesion: 300

window_size_list = [50, 100, 200] #suggesion: 8

max_length_of_document_vector_list = [350] #np.max([len(i.split()) for i in X_train_series]) #np.quantile([len(i.split()) for i in X_train_series], 0.7) ; suggesion: 8

embedding_only = False
save_results = True
test_size = 0.1 #suggestion: 0.1
training_set = "undersample" # "oversample", "undersample", "heterodox", "samequality" ; suggestion: oversample
embedding_set = False # "oversample", "undersample", "heterodox", "samequality", False ; suggestion: False

small_model = True
batch_size_list = [64, 128]
bert_epochs_list = [2]

results_file_name = "bert_results"
############################################

parameters = """PARAMETERS:
input_file_name = """ + input_file_name + """
max_length_of_document_vector_list = """ + str(max_length_of_document_vector_list) + """
save_results = """ + str(save_results) + """
test_size = """ + str(test_size) + """
training_set = """ + str(training_set) + """
small_model = """ + str(small_model) + """
batch_size_list = """ + str(batch_size_list) + """
bert_epochs_list = """ + str(bert_epochs_list)

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

logger.info(parameters)


from Utils import utils_ortho_hetero as fcts

'''
LOAD DATA
'''
logger.info("LOAD DATA")
if __name__ == "__main__":
    dtf = fcts.load_data(data_path = data_path,
                          input_file_name = input_file_name,
                          input_file_size = input_file_size,
                          input_file_type = input_file_type,
                          sample_size = "all")

if plot == 1 or plot == 2:
    fig, ax = plt.subplots()
    fig.suptitle("Label Distribution in Original Data", fontsize=12)
    dtf[label_field].reset_index().groupby(label_field).count().sort_values(by="index").plot(kind="barh", legend=False, ax=ax).grid(axis='x')
    plt.show()





'''
TRAIN TEST SPLIT
'''
logger.info("TRAIN TEST SPLIT")
dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=test_size)



#balanbce dataset
logger.info("BALANCE TRAINING SET")



if training_set == "oversample":
    over_sampler = RandomOverSampler(random_state=42)
    X_train, y_train = over_sampler.fit_resample(pd.DataFrame({"X": dtf_train[text_field]}), pd.DataFrame({"y":dtf_train[label_field]}))

elif training_set == "undersample":
    under_sampler = RandomUnderSampler(random_state=42)
    X_train, y_train = under_sampler.fit_resample(pd.DataFrame({"X": dtf_train[text_field]}), pd.DataFrame({"y":dtf_train[label_field]}))

else:
    X_train = pd.DataFrame({"X": dtf_train[text_field]})
    y_train = pd.DataFrame({"y": dtf_train[label_field]})





'''
if embedding_set == "oversample":
    over_sampler = RandomOverSampler(random_state=42)
    X_embed, y_embed = over_sampler.fit_resample(pd.DataFrame({"X": dtf_train[text_field]}), pd.DataFrame({"y":dtf_train[label_field]}))

elif embedding_set == "undersample":
    under_sampler = RandomUnderSampler(random_state=42)
    X_embed, y_embed = under_sampler.fit_resample(pd.DataFrame({"X": dtf_train[text_field]}), pd.DataFrame({"y":dtf_train[label_field]}))

elif embedding_set == "heterodox":
    X_embed = pd.DataFrame({"X": dtf_train.loc[dtf_train[label_field] == "heterodox"][text_field]})
    y_embed = pd.DataFrame({"y": dtf_train.loc[dtf_train[label_field] == "heterodox"][label_field]})

elif embedding_set == "samequality":
    X_embed = pd.DataFrame({"X": dtf_train.loc[dtf_train[label_field] == "samequality"][text_field]})
    y_embed = pd.DataFrame({"y": dtf_train.loc[dtf_train[label_field] == "samequality"][label_field]})

else:
    X_embed = pd.DataFrame({"X": dtf_train[text_field]})
    y_embed = pd.DataFrame({"y": dtf_train[label_field]})
'''


X_train_series = X_train.squeeze(axis=1)
y_train_series = y_train.squeeze(axis=1)

'''
X_embed_series = X_embed.squeeze(axis=1)
y_embed_series = y_embed.squeeze(axis=1)
'''


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





tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

text_lst = [text[:-50] for text in X_train["X"]]
text_lst = [' '.join(text.split()[:400]) for text in text_lst]

subtitles = ["design", "methodology", "approach", "originality", "value", "limitations", "implications"]

text_lst = [word for word in text_lst if word not in subtitles]

#text_lst = [text for text in text_lst if text]

corpus = text_lst

max_length_of_document_vector_loop = 0

for max_length_of_document_vector in max_length_of_document_vector_list:

    max_length_of_document_vector_loop += 1

    logger.info("max_length_of_document_vector_loop Nr: " + str(max_length_of_document_vector_loop))
    logger.info("max_length_of_document_vector : " + str(max_length_of_document_vector))


    ## add special tokens
    maxqnans = np.int((max_length_of_document_vector - 20) / 2)
    corpus_tokenized = ["[CLS] " +
                        " ".join(tokenizer.tokenize(re.sub(r'[^\w\s]+|\n', '',str(txt).lower().strip()))[:maxqnans]) +
                        " [SEP] " for txt in corpus]

    ## generate masks
    masks = [[1] * len(txt.split(" ")) + [0] * (max_length_of_document_vector - len(txt.split(" "))) for txt in corpus_tokenized]

    ## padding
    txt2seq = [txt + " [PAD]" * (max_length_of_document_vector - len(txt.split(" "))) if len(txt.split(" ")) != max_length_of_document_vector else txt for txt in corpus_tokenized]

    ## generate idx
    idx = [tokenizer.encode(seq.split(" "), is_split_into_words=True) for seq in txt2seq]
    minlen = min([len(i) for i in idx])
    idx = [i[:max_length_of_document_vector] for i in idx]



    ## feature matrix

    if small_model:
        X_train = [np.asarray(idx, dtype='int32'), np.asarray(masks, dtype='int32')]

    else:

        ## generate segments
        segments = []
        for seq in txt2seq:
            temp, i = [], 0
            for token in seq.split(" "):
                temp.append(i)
                if token == "[SEP]":
                    i += 1
            segments.append(temp)

        X_train = [np.asarray(idx, dtype='int32'), np.asarray(masks, dtype='int32'), np.asarray(segments, dtype='int32')]





    ##CLASSIFIER


    if small_model:

        ##DISTIL-BERT

        ## inputs
        idx = layers.Input((max_length_of_document_vector), dtype="int32", name="input_idx")
        masks = layers.Input((max_length_of_document_vector), dtype="int32", name="input_masks")
        # segments = layers.Input((max_length_of_document_vector), dtype="int32", name="input_segments")

        ## pre-trained bert with config
        config = transformers.DistilBertConfig(dropout=0.2, attention_dropout=0.2)
        config.output_hidden_states = False

        nlp = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased', config=config)
        bert_out = nlp(idx, attention_mask=masks)[0]

        ## fine-tuning
        x = layers.GlobalAveragePooling1D()(bert_out)
        x = layers.Dense(64, activation="relu")(x)
        y_out = layers.Dense(len(np.unique(y_train)), activation='softmax')(x)

        ## compile
        model = models.Model([idx, masks], y_out)

        for layer in model.layers[:3]:
            layer.trainable = False

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['mse'])

        model.summary()

    else:
        ##BERT

        ## inputs
        idx = layers.Input((max_length_of_document_vector), dtype="int32", name="input_idx")
        masks = layers.Input((max_length_of_document_vector), dtype="int32", name="input_masks")
        segments = layers.Input((max_length_of_document_vector), dtype="int32", name="input_segments")

        ## pre-trained bert
        nlp = transformers.TFBertModel.from_pretrained("bert-base-uncased")
        bert_out = nlp([idx, masks, segments])

        sequence_out = bert_out.last_hidden_state
        pooled_out = bert_out.pooler_output

        ## fine-tuning
        x = layers.GlobalAveragePooling1D() (sequence_out)
        x = layers.Dense(64, activation="relu")(x)
        y_out = layers.Dense(len(np.unique(y_train)), activation='softmax')(x)

        ## compile
        model = models.Model([idx, masks, segments], y_out)

        for layer in model.layers[:4]:
            layer.trainable = False

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['mse'])

        model.summary()







    # Feature engineer Test set

    text_lst = [text[:-50] for text in dtf_test[text_field_clean]]
    text_lst = [' '.join(text.split()[:400]) for text in text_lst]
    #text_lst = [text for text in text_lst if text]

    corpus = text_lst

    ## add special tokens
    maxqnans = np.int((max_length_of_document_vector - 20) / 2)
    corpus_tokenized = ["[CLS] " +
                        " ".join(tokenizer.tokenize(re.sub(r'[^\w\s]+|\n', '',str(txt).lower().strip()))[:maxqnans]) +
                        " [SEP] " for txt in corpus]

    ## generate masks
    masks = [[1] * len(txt.split(" ")) + [0] * (max_length_of_document_vector - len(txt.split(" "))) for txt in corpus_tokenized]

    ## padding
    txt2seq = [txt + " [PAD]" * (max_length_of_document_vector - len(txt.split(" "))) if len(txt.split(" ")) != max_length_of_document_vector_list else txt for txt in corpus_tokenized]

    ## generate idx
    idx = [tokenizer.encode(seq.split(" "), is_split_into_words=True) for seq in txt2seq]
    minlen = min([len(i) for i in idx])
    idx = [i[:max_length_of_document_vector] for i in idx]



    ## feature matrix

    if small_model:
        X_test = [np.array(idx, dtype='int32'), np.array(masks, dtype='int32')]

    else:
        ## generate segments
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
    dic_y_mapping = {n:label for n,label in enumerate(np.unique(y_train))}
    inverse_dic = {v:k for k,v in dic_y_mapping.items()}
    y_train_bin = np.array([inverse_dic[y] for y in y_train["y"]])

    batch_size_loop = 0

    for batch_size in batch_size_list:

        batch_size_loop += 1

        bert_epochs_loop = 0

        for bert_epochs in bert_epochs_list:

            class_time_start = time.perf_counter()

            bert_epochs_loop += 1

            logger.info("max_length_of_document_vector_loop Nr: " + str(max_length_of_document_vector_loop))
            logger.info("max_length_of_document_vector : " + str(max_length_of_document_vector))
            logger.info("batch_size Nr: " + str(batch_size_loop))
            logger.info("batch_size : " + str(batch_size))
            logger.info("bert_epochs_loop Nr: " + str(bert_epochs_loop))
            logger.info("bert_epochs : " + str(bert_epochs))

            ## train
            training = model.fit(x=X_train, y=y_train_bin, batch_size=batch_size, epochs=bert_epochs, shuffle=True, verbose=1, validation_split=0.3)

            ## test
            predicted_prob = model.predict(X_test)
            predicted = [dic_y_mapping[np.argmax(pred)] for pred in predicted_prob]

            y_test = dtf_test[label_field].values

            logger.info("TEST CALC")
            predicted_bin = [np.argmax(pred) for pred in predicted_prob]
            logger.info("TEST CALC FINISHED")

            y_test_bin = np.array([inverse_dic[y] for y in y_test])
            classes = np.array([dic_y_mapping[0], dic_y_mapping[1]])

            cm = metrics.confusion_matrix(y_test_bin, predicted_bin)
            logger.info("confusion matrix: " + str(cm))

            ''' PLAUSIBIBILITY CHECK
            y_test = y_train_vector.values.ravel()
            '''

            ## Accuracy, Precision, Recall
            logger.info("ACCURACY, PRECISION, RECALL")

            ## encode y

            auc = metrics.roc_auc_score(y_test_bin, predicted_prob[:, 1])
            precision, recall, threshold = metrics.precision_recall_curve(y_test_bin, predicted_prob[:, 1])
            auc_pr = metrics.auc(recall, precision)
            mse_negative = metrics.mean_squared_error(y_test_bin[y_test_bin == 0], predicted_prob[:, 1][y_test_bin == 0])
            mse_positive = metrics.mean_squared_error(y_test_bin[y_test_bin == 1], predicted_prob[:, 1][y_test_bin == 1])
            mcc = metrics.matthews_corrcoef(y_test_bin, predicted_bin)
            report = pd.DataFrame(metrics.classification_report(y_test, predicted, output_dict=True)).transpose()
            report.loc["auc"] = [auc] * len(report.columns)
            report.loc["auc_pr"] = [auc_pr] * len(report.columns)
            report.loc["mcc"] = [mcc] * len(report.columns)

            logger.info("Detail:")
            logger.info(report)

            class_time_total = time.perf_counter() - class_time_start

            logger.info(f"classification with {bert_epochs} epochs batch size {batch_size} for {len(dtf)} samples in {class_time_total} seconds")

            if save_results:
                logger.info("SAVING RESULTS")
                embedding_path = "numabs_" + str(len(dtf)) + "_embedlen_" + '_'.join(str(e) for e in embedding_vector_length_list) + "_embedepo_" + '_'.join(str(e) for e in num_epochs_for_embedding_list) + "_window_" + '_'.join(str(e) for e in window_size_list) + "_train_" + str(training_set) + "_embed_" + str(embedding_set) + "_testsize_" + str(test_size)

                results_path = data_path + "/results/" + str(results_file_name) + ".csv"

                now = time.asctime()
                result_id= int(time.time()*1000)

                result = pd.DataFrame({"time": [now],
                                       "result_id": [result_id],
                                       "length":[len(dtf)],
                                        "max_length_of_document_vector": [max_length_of_document_vector],
                                        "training_set": [training_set],
                                        "small_model": [small_model],
                                        "batch_size": [batch_size],
                                        "bert_epochs": [bert_epochs],
                                        "test_size": [test_size],
                                        "Negative_Label": [classes[0]],
                                        "Positive_Label": [classes[1]],
                                        "Support_Negative": [report["support"][classes[0]]],
                                        "Support_Positive": [report["support"][classes[1]]],
                                        "TN": [cm[0,0]],
                                        "FP": [cm[0,1]],
                                        "FN": [cm[1,0]],
                                        "TP": [cm[1,1]],
                                        "Precision_Negative": [report["precision"][classes[0]]],
                                        "Precision_Positive": [report["precision"][classes[1]]],
                                        "Recall_Negative": [report["recall"][classes[0]]],
                                        "Recall_Positive": [report["recall"][classes[1]]],
                                        "AUC": [auc],
                                        "AUC-PR": [auc_pr],
                                        "MCC": [mcc],
                                       "MSE_NEGATIVE":[mse_negative],
                                       "MSE_POSITIVE":[mse_positive],
                                       "MSE_AVERAGE":[(mse_negative+mse_positive)/2],
                                       "duration": [class_time_total]})

                results = pd.read_csv(results_path)
                results = pd.concat([results, result])
                results.to_csv(results_path, index=False)

                pd.DataFrame({"probabilities": predicted_prob[:,1]}).to_csv(data_path + "/pred_prob/" + str(result_id) + ".csv")





toc = time.perf_counter()
logger.info(f"whole script for {len(dtf)} in {toc-tic} seconds")
print("the end")
