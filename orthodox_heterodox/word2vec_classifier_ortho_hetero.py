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
import matplotlib as mpl
mpl.use('TkAgg')
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
input_file_name = "WOS_lee_heterodox_und_samequality_preprocessed_30000"
input_file_size = "all" #10000 #"all"
input_file_type = "csv"
output_file_name = "WOS_lee_heterodox_und_samequality_preprocessed_wip"
sample_size = "all"  #input_file_size #10000 #"all"
text_field_clean = "text_clean"  # "title" #"abstract"
text_field = "text"
label_field = "y"
cores = mp.cpu_count()  #mp.cpu_count()  #2
save = False  # False #True
plot = 1 #0 = none, 1 = some, 2 = all
use_pretrained = True #True if pretrained model should be used ("glove-wiki-gigaword-300d"), False to train embeddings based on vocabulary of input files
num_epochs_for_embedding = 10 #number of epochs to train the word embeddings
num_epochs_for_classification= 13 #number of epochs to train the the classifier
############################################


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
'''
under_sampler = RandomUnderSampler(random_state=42)
X_train, y_train = under_sampler.fit_resample(pd.DataFrame({"X": dtf_train[text_field_clean]}), pd.DataFrame({"y":dtf_train[label_field]}))
'''

over_sampler = RandomOverSampler(random_state=42)
X_train, y_train = over_sampler.fit_resample(pd.DataFrame({"X": dtf_train[text_field_clean]}), pd.DataFrame({"y":dtf_train[label_field]}))
X_train_series = X_train.squeeze(axis=1)
y_train_series = y_train.squeeze(axis=1)

if plot == 1 or plot == 2:
    fig, ax = plt.subplots()
    fig.suptitle("Label Distribution in Training Data", fontsize=12)
    y_train_series.value_counts().plot(kind="barh", legend=False, ax=ax).grid(axis='x')
    plt.show()




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
if use_pretrained:
    pretrained_start = time.perf_counter()

    nlp = gensim_api.load("glove-wiki-gigaword-300")

    word = "bad"
    nlp[word].shape
    nlp.most_similar(word)

    ## word embedding
    tot_words = [word] + [tupla[0] for tupla in nlp.most_similar(word, topn=20)]
    X = nlp[tot_words]

    pretrained_end = time.perf_counter()
    pretrained_time = pretrained_end - pretrained_start
    logger.info(f"loading pretrained vectors in {pretrained_time} seconds")

else:
    train_embeddings_start = time.perf_counter()

    nlp = gensim.models.word2vec.Word2Vec(lst_corpus, vector_size=300, window=8, min_count=3, sg=1, epochs=num_epochs_for_embedding)


    nlp.wv[word].shape
    nlp.wv.most_similar(word)

    ## word embedding
    tot_words = [word] + [tupla[0] for tupla in nlp.wv.most_similar(word, topn=20)]
    X = nlp.wv[tot_words]

    train_embeddings_end = time.perf_counter()
    train_embeddings_time = train_embeddings_end - train_embeddings_start
    logger.info(f"loading pretrained vectors {len(dtf)} documents and {num_epochs_for_embedding} epochs in {train_embeddings_time} seconds")

logger.info("WORD EMBEDDING FINISHED")



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
maxlength = 100 #np.max([len(i.split()) for i in X_train_series])
logger.info(f"maxlength = {maxlength}")

X_train = kprocessing.sequence.pad_sequences(lst_text2seq,maxlen=maxlength, padding="post", truncating="post")

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
X_test = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=maxlength, padding="post", truncating="post")


logger.info("SET UP EMBEDDING MATRIX")

## start the matrix (length of vocabulary x vector size) with all 0s
embeddings = np.zeros((len(dic_vocabulary)+1, 300))

for word,idx in dic_vocabulary.items():
    ## update the row with vector
    try:
        embeddings[idx] = nlp[word]
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
x_in = layers.Input(shape=(maxlength,))

## embedding
x = layers.Embedding(input_dim=embeddings.shape[0],
                     output_dim=embeddings.shape[1],
                     weights=[embeddings],
                     input_length=maxlength, trainable=False)(x_in)

## apply attention
x = attention_layer(x, neurons=maxlength)

## 2 layers of bidirectional lstm
x = layers.Bidirectional(layers.LSTM(units=maxlength, dropout=0.2, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(units=maxlength, dropout=0.2))(x)

## final dense layers
x = layers.Dense(64, activation='relu')(x)
y_out = layers.Dense(3, activation='softmax')(x)

## compile
model = models.Model(x_in, y_out)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

if print_charts_tables:
    model.summary()


logger.info("ENCODING FEATURES")

## encode y
dic_y_mapping = {n:label for n,label in enumerate(np.unique(y_train))}
inverse_dic = {v:k for k,v in dic_y_mapping.items()}
y_train_bin = np.array([inverse_dic[y] for y in y_train['y']])

logger.info("TRAINING")

logger.info("STARTING TRAINING")
## train
train_start = time.perf_counter()
training = model.fit(x=X_train, y=y_train_bin, batch_size=256, epochs=num_epochs_for_classification, shuffle=True, verbose=0, validation_split=0.3)
train_end = time.perf_counter()
train_time = train_end-train_start
logger.info(f"training model for {len(dtf)} and {num_epochs_for_classification} epochs in {train_time} seconds")

if plot == 1 or plot == 2:
    ## plot loss and accuracy
    metrics_plot = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
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
    plt.show()


## test
logger.info("TEST CALC")
predicted_prob = model.predict(X_test)
predicted = [dic_y_mapping[np.argmax(pred)] for pred in predicted_prob]
logger.info("TEST CALC FINISHED")


y_test = dtf_test[label_field].values
classes = np.unique(y_test)

if plot == 1 or plot == 2:
    ## Plot confusion matrix
    cm = metrics.confusion_matrix(y_test, predicted)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)
    ax.set(xlabel="Pred", ylabel="True", xticklabels=classes, yticklabels=classes, title="Confusion matrix")
    plt.yticks(rotation=0)
    plt.show()

toc = time.perf_counter()
logger.info(f"whole script for {len(dtf)} in {toc-tic} seconds")