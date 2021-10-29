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

'''
from tensorflow.keras import backend as K

## for bert language model
import transformers
'''




############################################
logging_level = logging.INFO  # logging.DEBUG #logging.WARNING
print_charts_tables = True  # False #True
input_file_name = "WOS_lee_heterodox_und_samequality_preprocessed"
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
use_embeddings = True #if True a trained model needs to be selected below
#which_embeddings = "word2vec_numabstracts_79431_embeddinglength_300_epochs_30" #specify model to use here
embedding_folder = "embeddings"
train_new = False #if True new embeddings are trained
num_epochs_for_embedding_list = [5] #number of epochs to train the word embeddings
num_epochs_for_classification_list= [1,2,3] #number of epochs to train the the classifier
embedding_vector_length_list = [300]
window_size_list = [4]
max_length_of_document_vector = 100 #np.max([len(i.split()) for i in X_train_series]) #np.quantile([len(i.split()) for i in X_train_series], 0.7)
embedding_only = False
############################################

parameters = """PARAMETERS:
input_file_name = """ + input_file_name + """
use_gigaword = """ + str(use_gigaword) + """
use_embeddings = """ + str(cores) + """
embedding_folder = """ + str(embedding_folder) + """
train_new = """ + str(train_new) + """
num_epochs_for_embedding_list = """ + str(num_epochs_for_embedding_list) + """
num_epochs_for_classification_list = """ + str(num_epochs_for_classification_list) + """
embedding_vector_length_list = """ + str(embedding_vector_length_list) + """
window_size_list = """ + str(window_size_list) + """
max_length_of_document_vector = """ + str(max_length_of_document_vector) + """
embedding_only = """ + str(embedding_only)






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

results_file = pd.DataFrame({"training_length":[],
                             "use_gigaword":[],
                             "num_epochs_for_embedding":[],
                             "num_epochs_for_classification":[],
                             "embedding_vector_length":[],
                             "window_size":[],
                             "max_length_of_document_vector":[],
                             "Negative_Label":[],
                             "Positive_Label":[],
                             "Support_Negative":[],
                             "Support_Positive":[],
                             "TN":[],
                             "FP":[],
                             "FN":[],
                             "TP":[],
                             "Precision_False":[],
                             "Precision_True":[],
                             "Recall_False":[],
                             "Recall_True":[],
                             "AUC":[],
                             "AUC-PR":[]})


if use_gigaword + use_embeddings + train_new != 1:
    sys.exit("invalid parameter setting: set only of use_gigaword, use_embeddings and which_embeddings to 'True' and the other two to 'False'")

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
dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=0.3)



#balanbce dataset
logger.info("BALANCE TRAINING SET")

under_sampler = RandomUnderSampler(random_state=42)
X_train, y_train = under_sampler.fit_resample(pd.DataFrame({"X": dtf_train[text_field_clean]}), pd.DataFrame({"y":dtf_train[label_field]}))

'''
over_sampler = RandomOverSampler(random_state=42)
X_train, y_train = over_sampler.fit_resample(pd.DataFrame({"X": dtf_train[text_field_clean]}), pd.DataFrame({"y":dtf_train[label_field]}))
'''

X_train_series = X_train.squeeze(axis=1)
y_train_series = y_train.squeeze(axis=1)

if plot == 1 or plot == 2:
    fig, ax = plt.subplots()
    fig.suptitle("Label Distribution in Training Data", fontsize=12)
    y_train_series.value_counts().plot(kind="barh", legend=False, ax=ax).grid(axis='x')
    plt.show()






for num_epochs_for_embedding in num_epochs_for_embedding_list:
    for embedding_vector_length in embedding_vector_length_list:
        for window_size in window_size_list:

            loop_params = """LOOP PARAMETERS:
            num_epochs_for_embedding = """ + str(num_epochs_for_embedding) + """
            embedding_vector_length = """ + str(embedding_vector_length) + """
            window_size = """ + str(window_size)

            logger.info(loop_params)

            #FEATURE ENGINEERING

            logger.info("FEATURE ENGINEERING")

            #pretrained = gensim.downloader.load("glove-wiki-gigaword-300")



            logger.info("FEATURE ENGINEERING FOR TRAINING SET")

            corpus = X_train_series


            ## create list of lists of unigrams
            lst_corpus = []
            for string in corpus:
               lst_words = string.split()
               lst_grams = [' '.join(lst_words[i:i+1]) for i in range(0, len(lst_words), 1)]
               lst_corpus.append(lst_grams)


            ## detect bigrams and trigrams
            bigrams_detector = gensim.models.phrases.Phrases(lst_corpus, delimiter=' ', min_count=5, threshold=10)
            bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
            trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[lst_corpus], delimiter=" ", min_count=2, threshold=10)
            trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)


            logger.info("STARTING WORD EMBEDDING")

            ## fit w2v
            if use_gigaword:
                gigaword_start = time.perf_counter()

                modelname_raw = "glove-wiki-gigaword-" + str(embedding_vector_length)

                modelname = "glove-wiki-gigaword-" + str(embedding_vector_length) + "_numabstracts_" + str(len(dtf))

                pretrained_vectors = modelname_raw

                nlp = gensim_api.load(pretrained_vectors)

                word = "bad"
                nlp[word].shape
                nlp.most_similar(word)

                ## word embedding
                tot_words = [word] + [tupla[0] for tupla in nlp.most_similar(word, topn=20)]
                X = nlp[tot_words]

                gigaword_end = time.perf_counter()
                gigaword_time = gigaword_end - gigaword_start
                logger.info(f"loading gigaword vectors in {gigaword_time} seconds")

            if use_embeddings:
                load_embeddings_start = time.perf_counter()

                modelname = "word2vec_numabstracts_" + str(len(dtf)) + "_embeddinglength_" + str(embedding_vector_length) + "_embeddingepochs_" + str(num_epochs_for_embedding) + "_window_" + str(window_size)

                pretrained_vectors = str(data_path) + "/" + embedding_folder + "/" + modelname

                nlp = gensim.models.word2vec.Word2Vec.load(pretrained_vectors)

                word = "bad"
                logger.info(str(nlp.wv[word].shape))
                logger.info(word + ": " + str(nlp.wv.most_similar(word)))

                ## word embedding
                tot_words = [word] + [tupla[0] for tupla in nlp.wv.most_similar(word, topn=20)]
                X = nlp.wv[tot_words]

                load_embeddings_end = time.perf_counter()
                load_embeddings_time = load_embeddings_end - load_embeddings_start
                logger.info(f"loading embeddings in {load_embeddings_time} seconds")


            if train_new:
                train_embeddings_start = time.perf_counter()

                modelname_raw = "word2vec_numabstracts_" + str(len(dtf)) + "_embeddinglength_" + str(embedding_vector_length) + "_embeddingepochs_" + str(num_epochs_for_embedding) + "_window_" + str(window_size)

                modelname = "newembedding" + str(modelname_raw)

                nlp = gensim.models.word2vec.Word2Vec(lst_corpus, vector_size=embedding_vector_length, window=window_size, sg=1, epochs=num_epochs_for_embedding, workers = cores, callbacks = None)

                nlp.save(str(data_path) + "/" + embedding_folder + "/" + modelname_raw)

                word = "bad"
                nlp.wv[word].shape
                nlp.wv.most_similar(word)

                ## word embedding
                tot_words = [word] + [tupla[0] for tupla in nlp.wv.most_similar(word, topn=20)]
                X = nlp.wv[tot_words]

                train_embeddings_end = time.perf_counter()
                train_embeddings_time = train_embeddings_end - train_embeddings_start
                logger.info(f"training word2vec for {len(dtf)} documents and {num_epochs_for_embedding} epochs in {train_embeddings_time} seconds")

            logger.info("WORD EMBEDDING FINISHED")






            if embedding_only == False:

                ## pca to reduce dimensionality from 300 to 3
                pca = manifold.TSNE(perplexity=40, n_components=3, init='pca')
                X = pca.fit_transform(X)

                ## create dtf
                dtf_ = pd.DataFrame(X, index=tot_words, columns=["x","y","z"])
                dtf_["input"] = 0
                dtf_["input"].iloc[0:1] = 1

                if plot == 2:
                    ## plot 3d
                    from mpl_toolkits.mplot3d import Axes3D
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(dtf_[dtf_["input"]==0]['x'],
                               dtf_[dtf_["input"]==0]['y'],
                               dtf_[dtf_["input"]==0]['z'], c="black")
                    ax.scatter(dtf_[dtf_["input"]==1]['x'],
                               dtf_[dtf_["input"]==1]['y'],
                               dtf_[dtf_["input"]==1]['z'], c="red")
                    ax.set(xlabel=None, ylabel=None, zlabel=None, xticklabels=[],
                           yticklabels=[], zticklabels=[])
                    for label, row in dtf_[["x","y","z"]].iterrows():
                        x, y, z = row
                        ax.text(x, y, z, s=label)


                ## tokenize text
                tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ', oov_token="NaN", filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
                tokenizer.fit_on_texts(lst_corpus)
                dic_vocabulary = tokenizer.word_index

                ## create sequence
                lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)

                ## padding sequence
                logger.info(f"max_length_of_document_vector = {max_length_of_document_vector}")

                X_train = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=max_length_of_document_vector, padding="post", truncating="post")

                i = 0

                ## list of text: ["I like this", ...]
                len_txt = len(X_train_series.iloc[i].split())
                if print_charts_tables:
                    logger.info(f"from: {X_train_series.iloc[i]} |len: {len_txt}")

                ## sequence of token ids: [[1, 2, 3], ...]
                len_tokens = len(X_train[i])
                if print_charts_tables:
                    logger.info(f"to: {X_train[i]} | len: {len(X_train[i])}")

                ## vocabulary: {"I":1, "like":2, "this":3, ...}
                if print_charts_tables:
                    logger.info(f"check: {X_train_series.iloc[i].split()[0]}  -- idx in vocabulary --> {dic_vocabulary[X_train_series.iloc[i].split()[0]]}")

                logger.info("FEATURE ENGINEERING FOR TEST SET")

                corpus = dtf_test["text_clean"]


                ## create list of n-grams
                lst_corpus = []
                for string in corpus:
                    lst_words = string.split()
                    lst_grams = [" ".join(lst_words[i:i+1]) for i in range(0, len(lst_words), 1)]
                    lst_corpus.append(lst_grams)

                ## detect common bigrams and trigrams using the fitted detectors
                lst_corpus = list(bigrams_detector[lst_corpus])
                lst_corpus = list(trigrams_detector[lst_corpus])

                ## text to sequence with the fitted tokenizer
                lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)

                ## padding sequence
                X_test = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=max_length_of_document_vector, padding="post", truncating="post")


                logger.info("SET UP EMBEDDING MATRIX")

                ## start the matrix (length of vocabulary x vector size) with all 0s
                embeddings = np.zeros((len(dic_vocabulary)+1, embedding_vector_length))

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


                logger.info("NETWORK ARCHITECTURE")

                ## code attention layer
                def attention_layer(inputs, neurons):
                    x = layers.Permute((2,1))(inputs)
                    x = layers.Dense(neurons, activation="softmax")(x)
                    x = layers.Permute((2,1), name="attention")(x)
                    x = layers.multiply([inputs, x])
                    return x

                ## input
                x_in = layers.Input(shape=(max_length_of_document_vector,))

                ## embedding
                x = layers.Embedding(input_dim=embeddings.shape[0],
                                     output_dim=embeddings.shape[1],
                                     weights=[embeddings],
                                     input_length=max_length_of_document_vector, trainable=False)(x_in)

                ## apply attention
                x = attention_layer(x, neurons=max_length_of_document_vector)

                ## 2 layers of bidirectional lstm
                x = layers.Bidirectional(layers.LSTM(units=max_length_of_document_vector, dropout=0.2, return_sequences=True))(x)
                x = layers.Bidirectional(layers.LSTM(units=max_length_of_document_vector, dropout=0.2))(x)

                ## final dense layers
                x = layers.Dense(64, activation='relu')(x)
                y_out = layers.Dense(2, activation='softmax')(x)

                ## compile
                model = models.Model(x_in, y_out)
                model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

                if print_charts_tables:
                    model.summary()

                logger.info("ENCODING FEATURES")

                random_test_labels = False

                if random_test_labels:

                    newvec = []
                    uniquelabels = np.unique(y_train)
                    for i in range(len(y_train)):
                        newvec.append(random.choice(uniquelabels))

                    y_train = pd.DataFrame({"y": newvec})

                ## encode y
                dic_y_mapping = {n:label for n,label in enumerate(np.unique(y_train))}
                dic_y_mapping = {0: "samequality", 1:"heterodox"}
                inverse_dic = {v:k for k,v in dic_y_mapping.items()}
                y_train_bin = np.array([inverse_dic[y] for y in y_train['y']])






                for num_epochs_for_classification in num_epochs_for_classification_list:

                    logger.info("TRAINING")

                    logger.info("EPOCHS: " + str(num_epochs_for_classification))

                    logger.info("STARTING TRAINING")
                    ## train
                    train_start = time.perf_counter()
                    training = model.fit(x=X_train, y=y_train_bin, batch_size=256, epochs=num_epochs_for_classification, shuffle=True, verbose=0, validation_split=0.3, workers=cores)
                    train_end = time.perf_counter()
                    train_time = train_end-train_start
                    logger.info(f"training model for {len(dtf)} and {num_epochs_for_classification} epochs in {train_time} seconds")


                    ## plot loss and accuracy
                    metrics_plot = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]


                    if plot == 1 or plot == 2:
                        fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)

                        ax[0].set(title="Training")
                        ax11 = ax[0].twinx()
                        ax[0].plot(training.history['loss'], color='black')
                        ax[0].set_xlabel('Epochs')
                        ax[0].set_ylabel('Loss', color='black')

                        for metric in metrics_plot:
                            ax11.plot(training.history[metric], label=metric)
                        ax11.set_ylabel("Score", color='steelblue')
                        ax11.legend()

                        ax[1].set(title="Validation")
                        ax22 = ax[1].twinx()
                        ax[1].plot(training.history['val_loss'], color='black')
                        ax[1].set_xlabel('Epochs')
                        ax[1].set_ylabel('Loss', color='black')
                        for metric in metrics_plot:
                             ax22.plot(training.history['val_'+metric], label=metric)
                        ax22.set_ylabel("Score", color="steelblue")

                        training_plot_path = data_path + "/results/w2v_training_plot_" + str(modelname) + "_classificatinepochs_" + str(num_epochs_for_classification) + "_maxabstractlength_" + str(max_length_of_document_vector) + ".png"

                        plt.savefig(training_plot_path, bbox_inches='tight')


                    ## test
                    logger.info("TEST CALC")
                    predicted_prob = model.predict(X_test)
                    predicted = [dic_y_mapping[np.argmax(pred)] for pred in predicted_prob]
                    logger.info("TEST CALC FINISHED")


                    y_test = dtf_test[label_field].values
                    classes = np.unique(y_test)

                    cm = metrics.confusion_matrix(y_test, predicted)
                    logger.info("confusion matrix: " + str(cm))

                    '''
                    cm_path = data_path + "/results/w2v_cm_" + str(modelname) + "_classificatinepochs_" + str(num_epochs_for_classification) + "_maxabstractlength_" + str(max_length_of_document_vector) + ".csv"
                    np.savetxt(cm_path, cm, delimiter=",", fmt='%1.0f')
                    '''


                    ## Accuracy, Precision, Recall
                    logger.info("ACCURACY, PRECISION, RECALL")

                    ## encode y
                    y_test_bin = np.array([inverse_dic[y] for y in y_test])

                    auc = metrics.roc_auc_score(y_test, predicted_prob[:, 1])
                    precision, recall, threshold = metrics.precision_recall_curve(y_test_bin, predicted_prob[:, 1])
                    auc_pr = metrics.auc(recall, precision)
                    report = pd.DataFrame(metrics.classification_report(y_test, predicted,output_dict=True)).transpose()
                    report.loc["auc"] = [auc]*len(report.columns)
                    report.loc["auc_pr"] = [auc_pr] * len(report.columns)

                    logger.info("Detail:")
                    logger.info(report)
                    '''
                    report_path = data_path + "/results/w2v_report_" + str(modelname) + "_classificatinepochs_" + str(num_epochs_for_classification) + "_maxabstractlength_" + str(max_length_of_document_vector) + ".csv"
                    report.to_csv(report_path)
                    '''

                    if plot == 1 or plot == 2:
                        ## Plot confusion matrix
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)
                        ax.set(xlabel="Pred", ylabel="True", xticklabels=classes, yticklabels=classes, title="Confusion matrix")
                        plt.yticks(rotation=0)
                        plt.savefig(data_path +"/results/word2vec_ortho_hetero.png", bbox_inches='tight')


                    results_file = results_file.append(pd.DataFrame({"training_length": [len(y_train)],
                                                                     "use_gigaword": [use_gigaword],
                                                                     "num_epochs_for_embedding": [num_epochs_for_embedding],
                                                                     "num_epochs_for_classification": [num_epochs_for_classification],
                                                                     "embedding_vector_length": [embedding_vector_length],
                                                                     "window_size": [window_size],
                                                                     "max_length_of_document_vector": [max_length_of_document_vector],
                                                                     "Negative_Label": [classes[0]],
                                                                     "Positive_Label": [classes[1]],
                                                                     "Support_Negative": [report["support"][classes[0]]],
                                                                     "Support_Positive": [report["support"][classes[1]]],
                                                                     "TN": [cm[0,0]],
                                                                     "FP": [cm[0,1]],
                                                                     "FN": [cm[1,0]],
                                                                     "TP": [cm[1,1]],
                                                                     "Precision_False": [report["precision"][classes[0]]],
                                                                     "Precision_True": [report["precision"][classes[1]]],
                                                                     "Recall_False": [report["recall"][classes[0]]],
                                                                     "Recall_True": [report["recall"][classes[1]]],
                                                                     "AUC": [auc],
                                                                     "AUC-PR": [auc_pr]}))


results_path = data_path + "/results/w2v_results_numabstracts_" + str(len(dtf)) + ".csv"

results_file.to_csv(results_path)

toc = time.perf_counter()
logger.info(f"whole script for {len(dtf)} in {toc-tic} seconds")
print("the end")