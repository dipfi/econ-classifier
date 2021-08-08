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

'''
## for deep learning
from tensorflow.keras import models, layers, preprocessing as kprocessing
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
plot = 1 #0 = none, 1 = some, 2 = all
sentiment = False
max_features = 1000
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
LENGTH ANALYSIS
'''

logger.info("LENGTH ANALYSIS")

dtf['word_count'] = dtf[text_field].progress_apply(lambda x: len(str(x).split(" ")))

dtf['char_count'] = dtf[text_field].progress_apply(lambda x: sum(len(word) for word in str(x).split(" ")))

dtf['sentence_count'] = dtf[text_field].progress_apply(lambda x: len(str(x).split(".")))

dtf['avg_word_length'] = dtf['char_count'] / dtf['word_count']

dtf['avg_sentence_lenght'] = dtf['word_count'] / dtf['sentence_count']

dtf.groupby(by="y")[["avg_word_length","avg_sentence_lenght", "sentence_count", "word_count"]].mean()

minimum_sample = min(dtf.groupby(by="y").size())


if plot == 2:
    x, y = "avg_word_length", "y"
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(x, fontsize=12)
    for i in dtf[y].unique():
        sns.distplot(dtf[dtf[y] == i][x], hist=False, kde=True,
                     kde_kws={"shade": True}, axlabel="density")
    ax.legend(dtf[y].unique())
    plt.show()

    x, y = "avg_sentence_lenght", "y"
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(x, fontsize=12)
    for i in dtf[y].unique():
        sns.distplot(dtf[dtf[y] == i][x], hist=False, kde=True,
                     kde_kws={"shade": True}, axlabel="density")
    ax.legend(dtf[y].unique())
    plt.show()

    x, y = "sentence_count", "y"
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(x, fontsize=12)
    for i in dtf[y].unique():
        sns.distplot(dtf[dtf[y] == i][x], hist=False, kde=True,
                     kde_kws={"shade": True}, axlabel="density")
    ax.legend(dtf[y].unique())
    plt.show()

    x, y = "word_count", "y"
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(x, fontsize=12)
    for i in dtf[y].unique():
        sns.distplot(dtf[dtf[y] == i][x], hist=False, kde=True,
                     kde_kws={"shade": True}, axlabel="density")
    ax.legend(dtf[y].unique())
    plt.show()




'''
SENTIMENT ANALYSIS
'''

if sentiment == True:
    logger.info("SENTIMENT ANALYSIS")
    from textblob import TextBlob

    print("SENTIMENT ANALYSIS")

    dtf["polarity"] = dtf["text_clean"].progress_apply(lambda x: TextBlob(x).sentiment.polarity) #-1 = negative, +1=positive
    dtf["subjectivity"] = dtf["text_clean"].progress_apply(lambda x: TextBlob(x).sentiment.subjectivity) #0 = objective, 1=subjective
    dtf.groupby(by="y")[["polarity", "subjectivity"]].mean()

    if plot == 2:
        x, y = "polarity", "y"
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.suptitle(x, fontsize=12)
        for i in dtf[y].unique():
            sns.distplot(dtf[dtf[y] == i][x], hist=False, kde=True,
                         kde_kws={"shade": True}, axlabel="density")
        ax.legend(dtf[y].unique())
        plt.show()

        x, y = "subjectivity", "y"
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.suptitle(x, fontsize=12)
        for i in dtf[y].unique():
            sns.distplot(dtf[dtf[y] == i][x], hist=False, kde=True,
                         kde_kws={"shade": True}, axlabel="density")
        ax.legend(dtf[y].unique())
        plt.show()


    dtf_length_sentiment = dtf.copy()

    ### Starting point
    dtf = dtf_length_sentiment.copy()



toc = time.perf_counter()
print(f" LENGTH & SENTIMENT ANALYSIS {input_file_name} --> {sample_size} in {toc-tic} seconds")


dtf_length = dtf.copy()

### Starting point
dtf = dtf_length.copy()



'''
TRAIN TEST SPLIT
'''
logger.info("TRAIN TEST SPLIT")
'''
all_journals = dtf[["Source Title", label_field]].drop_duplicates().copy()
heterodox_test = all_journals["Source Title"][all_journals[label_field] == "heterodox"].sample(frac=0.3).copy()
samequality_test = all_journals["Source Title"][all_journals[label_field] == "samequality"].sample(frac=0.3).copy()
all_test = heterodox_test.append(samequality_test)

logger.info("Heterodox Test Journals: " + heterodox_test)
logger.info("Samequality Test Journals: " + samequality_test)

## split dataset
dtf_test = dtf[dtf["Source Title"].isin(all_test)]
dtf_train = dtf[~dtf["Source Title"].isin(all_test)]

dtf_train = dtf_train[[text_field_clean, label_field]]
dtf_test = dtf_test[[text_field_clean, label_field]]
'''


dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=0.3)



#balanbce dataset
logger.info("BALANCE TRAINING SET")
'''
under_sampler = RandomUnderSampler(random_state=42)
X_train, y_train = under_sampler.fit_resample(pd.DataFrame({"X": dtf_train[text_field_clean]}), pd.DataFrame({"y":dtf_train[label_field]}))
'''

over_sampler = RandomOverSampler(random_state=42)
X_train, y_train = over_sampler.fit_resample(pd.DataFrame({"X": dtf_train[text_field_clean]}), pd.DataFrame({"y":dtf_train[label_field]}))
X_train_vector = X_train
y_train_vector = y_train

if plot == 1 or plot == 2:
    fig, ax = plt.subplots()
    fig.suptitle("Label Distribution in Training Data", fontsize=12)
    y_train.value_counts().plot(kind="barh", legend=False, ax=ax).grid(axis='x')
    plt.show()


#FEATURE ENGINEERING
logger.info("FEATURE ENGINEERING")
## Count (classic BoW)
#vectorizer = feature_extraction.text.CountVectorizer(max_features=max_features, ngram_range=(1,2))

## Tf-Idf (advanced variant of BoW)
logger.info("TFIDF VECTORIZER")

vectorizer = feature_extraction.text.TfidfVectorizer(max_features=max_features, ngram_range=(1,2))

corpus = X_train["X"]

vectorizer.fit(corpus)
X_train = vectorizer.transform(corpus)
dic_vocabulary = vectorizer.vocabulary_

if plot == 2:
    sns.heatmap(X_train.todense()[:,np.random.randint(0,X_train.shape[1],100)]==0, vmin=0, vmax=1, cbar=False).set_title('Sparse Matrix Sample')




#FEATURE SELECTION
logger.info("FEATURE SELECTION")
y = y_train
X_names = vectorizer.get_feature_names()
p_value_limit = 0.95

dtf_features = pd.DataFrame()
for cat in np.unique(y):
    chi2, p = feature_selection.chi2(X_train, y==cat)
    dtf_features = dtf_features.append(pd.DataFrame(
                   {"feature":X_names, "score":1-p, "y":cat}))
    dtf_features = dtf_features.sort_values([label_field,"score"],
                    ascending=[True,False])
    dtf_features = dtf_features[dtf_features["score"]>p_value_limit]

X_names = dtf_features["feature"].unique().tolist()


'''
for cat in np.unique(y):
   print("# {}:".format(cat))
   print("  . selected features:",
         len(dtf_features[dtf_features["y"]==cat]))
   print("  . top features:", ",".join(
dtf_features[dtf_features["y"]==cat]["feature"].values[:10]))
   print(" ")
'''


#shorter
logger.info("SHORTENING VOCABULARY")
vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)

vectorizer.fit(corpus)
X_train = vectorizer.transform(corpus)
dic_vocabulary = vectorizer.vocabulary_


#classify
logger.info("SET UP CLASSIFIER")

classifier = naive_bayes.MultinomialNB()

## pipeline
model = pipeline.Pipeline([("vectorizer", vectorizer),
                           ("classifier", classifier)])

## train classifier
logger.info("TRAIN CLASSIFIER")

y_train = y_train.values.ravel()

model["classifier"].fit(X_train, y_train)

def get_salient_words(nb_clf, vect, class_ind, class_op):
    """Return salient words for given class
    Parameters
    ----------
    nb_clf : a Naive Bayes classifier (e.g. MultinomialNB, BernoulliNB)
    vect : CountVectorizer
    class_ind : int
    Returns
    -------
    list
        a sorted list of (word, log prob) sorted by log probability in descending order.
    """

    words = vect.get_feature_names()
    zipped = list(zip(words, nb_clf.feature_log_prob_[class_ind], nb_clf.feature_log_prob_[class_op], np.log(np.exp(nb_clf.feature_log_prob_[class_ind])/np.exp(nb_clf.feature_log_prob_[class_op]))))
    sorted_zip = sorted(zipped, key=lambda t: t[3], reverse=True)

    return sorted_zip

neg_salient_top_20 = get_salient_words(model["classifier"], vectorizer, 0, 1)[:20]
pos_salient_top_20 = get_salient_words(model["classifier"], vectorizer, 1, 0)[:20]

if plot == 1 or plot == 2:
    fig, ax = plt.subplots()
    fig.suptitle("Salient words for heterodox articles", fontsize=12)
    plt.bar([i[0] for i in neg_salient_top_20], [i[3] for i in neg_salient_top_20])
    plt.xticks(rotation='vertical')
    fig.subplots_adjust(bottom=0.4)
    plt.show()

    fig, ax = plt.subplots()
    fig.suptitle("Salient words for orthodox articles", fontsize=12)
    plt.bar([i[0] for i in pos_salient_top_20], [i[3] for i in pos_salient_top_20])
    plt.xticks(rotation='vertical')
    fig.subplots_adjust(bottom=0.4)
    plt.show()



## test
logger.info("TEST CLASSIFIER")

X_test = dtf_test[text_field_clean].values

''' PLAUSIBIBILITY CHECK
X_test = X_train_vector.values.ravel()
'''

predicted = model.predict(X_test)
predicted_prob = model.predict_proba(X_test)

y_test = dtf_test[label_field].values

''' PLAUSIBIBILITY CHECK
y_test = y_train_vector.values.ravel()
'''

classes = np.unique(y_test)
y_test_array = pd.get_dummies(y_test, drop_first=False).values


## Accuracy, Precision, Recall
logger.info("ACCURACY, PRECISION, RECALL")

accuracy = metrics.accuracy_score(y_test, predicted)
auc = metrics.roc_auc_score(y_test, predicted_prob[:,1], multi_class="ovr")
print("Accuracy:", round(accuracy, 2))
print("Auc:", round(auc, 2))
print("Detail:")
print(metrics.classification_report(y_test, predicted))

if plot == 1 or plot == 2:
    ## Plot confusion matrix
    cm = metrics.confusion_matrix(y_test, predicted)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,cbar=False)
    ax.set(xlabel="Pred", ylabel="True", xticklabels=classes,yticklabels=classes, title="Confusion matrix")
    plt.yticks(rotation=0)
    fig, ax = plt.subplots(nrows=1, ncols=2)

    ## Plot roc
    for i in range(len(classes)):
        fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:, i],predicted_prob[:, i])
        ax[0].plot(fpr, tpr, lw=3,label='{0} (area={1:0.2f})'.format(classes[i],metrics.auc(fpr, tpr)))
    ax[0].plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    ax[0].set(xlim=[-0.05, 1.0], ylim=[0.0, 1.05],
              xlabel='False Positive Rate',
              ylabel="True Positive Rate (Recall)",
              title="Receiver operating characteristic")
    ax[0].legend(loc="lower right")
    ax[0].grid(True)

    ## Plot precision-recall curve
    for i in range(len(classes)):
        precision, recall, thresholds = metrics.precision_recall_curve(y_test_array[:, i], predicted_prob[:, i])
        ax[1].plot(recall, precision, lw=3,label='{0} (area={1:0.2f})'.format(classes[i],metrics.auc(recall, precision)))
    ax[1].set(xlim=[0.0, 1.05], ylim=[0.0, 1.05], xlabel='Recall',ylabel="Precision", title="Precision-Recall curve")
    ax[1].legend(loc="best")
    ax[1].grid(True)
    plt.show()

## select observation
logger.info("INTERPRETATION WITH LIME")
for i in random.sample(range(1, len(dtf_test)), 10):
    txt_instance = dtf_test["text_clean"].iloc[i]

    if plot == 1 or plot == 2:
        ## check true value and predicted value
        print("True:", y_test[i], "--> Pred:", predicted[i], "| Prob:", round(np.max(predicted_prob[i]),2))

        import IPython
        ## show explanation
        explainer = lime_text.LimeTextExplainer(class_names=np.unique(y_train))
        explained = explainer.explain_instance(txt_instance, model.predict_proba, num_features=9, top_labels=1)
        explained.show_in_notebook(text=txt_instance, predict_proba=False)
        name_of_html = "lime_" + os.path.basename(__file__) + "_observation_" + str(i) + ".html"
        explained.save_to_file(data_path + "/lime_explanations/" + name_of_html)


'''
WORD FREQUENCY
'''
print("WORD FREQUENCY")

if plot == 2:
    top = 10
    y = "samequality"
    corpus = dtf[dtf["y"] == y]["text_clean"]
    lst_tokens = nltk.tokenize.word_tokenize(corpus.str.cat(sep=" "))
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.suptitle("Most frequent words samequality", fontsize=15)

    ## unigrams
    dic_words_freq = nltk.FreqDist(lst_tokens)
    dtf_uni = pd.DataFrame(dic_words_freq.most_common(), columns=["Word", "Freq"])
    dtf_uni.set_index("Word").iloc[:top, :].sort_values(by="Freq").plot(kind="barh", title="Unigrams", ax=ax[0], legend=False).grid(axis='x')
    ax[0].set(ylabel=None)

    ## bigrams
    dic_words_freq = nltk.FreqDist(nltk.ngrams(lst_tokens, 2))
    dtf_bi = pd.DataFrame(dic_words_freq.most_common(),columns=["Word", "Freq"])
    dtf_bi["Word"] = dtf_bi["Word"].apply(lambda x: " ".join(string for string in x))
    dtf_bi.set_index("Word").iloc[:top, :].sort_values(by="Freq").plot(kind="barh", title="Bigrams", ax=ax[1],legend=False).grid(axis='x')
    ax[1].set(ylabel=None)
    plt.show()



    top = 10
    y = "heterodox"
    corpus = dtf[dtf["y"] == y]["text_clean"]
    lst_tokens = nltk.tokenize.word_tokenize(corpus.str.cat(sep=" "))
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.suptitle("Most frequent words heterodox", fontsize=15)

    ## unigrams
    dic_words_freq = nltk.FreqDist(lst_tokens)
    dtf_uni = pd.DataFrame(dic_words_freq.most_common(), columns=["Word", "Freq"])
    dtf_uni.set_index("Word").iloc[:top, :].sort_values(by="Freq").plot(kind="barh", title="Unigrams", ax=ax[0], legend=False).grid(axis='x')
    ax[0].set(ylabel=None)

    ## bigrams
    dic_words_freq = nltk.FreqDist(nltk.ngrams(lst_tokens, 2))
    dtf_bi = pd.DataFrame(dic_words_freq.most_common(),columns=["Word", "Freq"])
    dtf_bi["Word"] = dtf_bi["Word"].apply(lambda x: " ".join(string for string in x))
    dtf_bi.set_index("Word").iloc[:top, :].sort_values(by="Freq").plot(kind="barh", title="Bigrams", ax=ax[1],legend=False).grid(axis='x')
    ax[1].set(ylabel=None)
    plt.show()

'''
wc = wordcloud.WordCloud(background_color='black', max_words=100, max_font_size=35)
wc = wc.generate(str(corpus))
fig = plt.figure(num=1)
plt.axis('off')
plt.imshow(wc, cmap=None)
plt.show()
'''

toc = time.perf_counter()
print(f"whole script for {len(dtf)} in {toc-tic} seconds")