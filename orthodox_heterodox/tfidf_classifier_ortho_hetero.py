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
plot = False
sentiment = False
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


if __name__ == "__main__":
    dtf = fcts.load_data(data_path = data_path,
                          input_file_name = input_file_name,
                          input_file_size = input_file_size,
                          input_file_type = input_file_type,
                          sample_size = "all")
print("huhu")



fig, ax = plt.subplots()
fig.suptitle(label_field, fontsize=12)
dtf[label_field].reset_index().groupby(label_field).count().sort_values(by="index").plot(kind="barh", legend=False, ax=ax).grid(axis='x')
plt.show()




'''
LENGTH ANALYSIS
'''

print("TLENGTH ANALYSIS")

dtf['word_count'] = dtf[text_field].progress_apply(lambda x: len(str(x).split(" ")))

dtf['char_count'] = dtf[text_field].progress_apply(lambda x: sum(len(word) for word in str(x).split(" ")))

dtf['sentence_count'] = dtf[text_field].progress_apply(lambda x: len(str(x).split(".")))

dtf['avg_word_length'] = dtf['char_count'] / dtf['word_count']

dtf['avg_sentence_lenght'] = dtf['word_count'] / dtf['sentence_count']

dtf.groupby(by="y")[["avg_word_length","avg_sentence_lenght", "sentence_count", "word_count"]].mean()

minimum_sample = min(dtf.groupby(by="y").size())

import matplotlib.pyplot as plt
import seaborn as sns

if plot == True:
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
## for sentiment
    from textblob import TextBlob

    print("SENTIMENT ANALYSIS")

    dtf["polarity"] = dtf["text_clean"].progress_apply(lambda x: TextBlob(x).sentiment.polarity) #-1 = negative, +1=positive
    dtf["subjectivity"] = dtf["text_clean"].progress_apply(lambda x: TextBlob(x).sentiment.subjectivity) #0 = objective, 1=subjective
    dtf.groupby(by="y")[["polarity", "subjectivity"]].mean()

    if plot == True:
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



'''
if save == True:
    if sample_size == len(data_orig):
        print("sample_size == len(data_orig)")
        dtf.to_csv(data_path + '/' + input_file_name + "_preproc.csv", index=False)

    else:
        dtf.to_csv(data_path + '/sample_for_damian_' + str(sample_size) + '_preproc.csv', index=False)
'''


toc = time.perf_counter()
print(f" LENGTH & SENTIMENT ANALYSIS {input_file_name} --> {sample_size} in {toc-tic} seconds")


dtf_length = dtf.copy()

### Starting point
dtf = dtf_length.copy()



'''
NAMED ENTITY RECOGNITION
'''
'''
print("NAMED ENTITY RECOGNITION")

## call model
ner = spacy.load("en_core_web_lg")

## tag text
txt = dtf["text"].iloc[0]
doc = ner(txt)

## display result
spacy.displacy.render(doc, style="ent")




## tag text and exctract tags into a list
dtf["tags"] = dtf["text"].apply(lambda x: [(tag.text, tag.label_) for tag in ner(x).ents] )

## utils function to count the element of a list
def utils_lst_count(lst):
    dic_counter = collections.Counter()
    for x in lst:
        dic_counter[x] += 1
    dic_counter = collections.OrderedDict(
                     sorted(dic_counter.items(),
                     key=lambda x: x[1], reverse=True))
    lst_count = [ {key:value} for key,value in dic_counter.items() ]
    return lst_count

## count tags
dtf["tags"] = dtf["tags"].apply(lambda x: utils_lst_count(x))

## utils function create new column for each tag category
def utils_ner_features(lst_dics_tuples, tag):
    if len(lst_dics_tuples) > 0:
        tag_type = []
        for dic_tuples in lst_dics_tuples:
            for tuple in dic_tuples:
                type, n = tuple[1], dic_tuples[tuple]
                tag_type = tag_type + [type]*n
                dic_counter = collections.Counter()
                for x in tag_type:
                    dic_counter[x] += 1
        return dic_counter[tag]
    else:
        return 0

## extract features
tags_set = []
for lst in dtf["tags"].tolist():
     for dic in lst:
          for k in dic.keys():
              tags_set.append(k[1])
tags_set = list(set(tags_set))
for feature in tags_set:
     dtf["tags_"+feature] = dtf["tags"].apply(lambda x: utils_ner_features(x, feature))

## print result
print(dtf.head())
'''

'''
top = 5
y = "ENTERTAINMENT"

tags_list = dtf[dtf["y"] == y]["tags"].sum()
map_lst = list(map(lambda x: list(x.keys())[0], tags_list))
dtf_tags = pd.DataFrame(map_lst, columns=['tag', 'type'])
dtf_tags["count"] = 1
dtf_tags = dtf_tags.groupby(['type', 'tag']).count().reset_index().sort_values("count", ascending=False)
fig, ax = plt.subplots()
fig.suptitle("Top frequent tags", fontsize=12)
sns.barplot(x="count", y="tag", hue="type", data=dtf_tags.iloc[:top, :], dodge=False, ax=ax)
ax.grid(axis="x")
plt.show()
'''

'''
## predict wit NER
txt = dtf["text"].iloc[0]
entities = ner(txt).ents

## tag text
tagged_txt = txt
for tag in entities:
    tagged_txt = re.sub(tag.text, "_".join(tag.text.split()), tagged_txt)

## show result
print(tagged_txt)




dtf_ner = dtf.copy()

### Starting point
dtf = dtf_ner.copy()
'''

'''
WORD FREQUENCY
'''
'''
print("WORD FREQUENCY")
top = 5
y = 1
corpus = dtf[dtf["y"] == y]["text_clean"]
lst_tokens = nltk.tokenize.word_tokenize(corpus.str.cat(sep=" "))
fig, ax = plt.subplots(nrows=1, ncols=2)
fig.suptitle("Most frequent words", fontsize=15)

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




lst_words = ["money", "institution", "society"]

## count
lst_grams = [len(word.split(" ")) for word in lst_words]
vectorizer = feature_extraction.text.CountVectorizer(vocabulary=lst_words,ngram_range=(min(lst_grams),max(lst_grams)))

dtf_X = pd.DataFrame(vectorizer.fit_transform(dtf["text_clean"]).todense(), columns=lst_words)

## add the new features as columns
dtf1 = pd.concat([dtf, dtf_X.set_index(dtf.index)], axis=1)
print(dtf1.head())



wc = wordcloud.WordCloud(background_color='black', max_words=100, max_font_size=35)
wc = wc.generate(str(corpus))
fig = plt.figure(num=1)
plt.axis('off')
plt.imshow(wc, cmap=None)
plt.show()






dtf_frequencies = dtf.copy()

### Starting point
dtf = dtf_frequencies.copy()
'''


'''
WORD VECTORS
'''
'''
print("WORD VECTORS")

nlp = gensim_api.load("glove-wiki-gigaword-300")

word = "love"

print(nlp[word])

print(nlp[word].shape)




## find closest vectors
labels, X, x, y = [], [], [], []
for t in nlp.most_similar(word, topn=20):
    X.append(nlp[t[0]])
    labels.append(t[0])

## reduce dimensions
pca = manifold.TSNE(perplexity=40, n_components=2, init='pca')
new_values = pca.fit_transform(X)
for value in new_values:
    x.append(value[0])
    y.append(value[1])

## plot
fig = plt.figure()
for i in range(len(x)):
    plt.scatter(x[i], y[i], c="black")
    plt.annotate(labels[i], xy=(x[i],y[i]), xytext=(5,2), textcoords='offset points', ha='right', va='bottom')

## add center
plt.scatter(x=0, y=0, c="red")
plt.annotate(word, xy=(0,0), xytext=(5,2), textcoords='offset points', ha='right', va='bottom')

dtf_vectors = dtf.copy()

### Starting point
dtf = dtf_vectors.copy()
'''



'''
TOPIC MODELING
'''
'''
print("TOPIC MODELING")

y = 0
corpus = dtf[dtf["y"] == y]["text_clean"]
## pre-process corpus
lst_corpus = []
for string in corpus:
    lst_words = string.split()
    lst_grams = [" ".join(lst_words[i:i + 2]) for i in range(0,len(lst_words), 2)]
    lst_corpus.append(lst_grams)  ## map words to an id
id2word = gensim.corpora.Dictionary(lst_corpus)  ## create dictionary word:freq
dic_corpus = [id2word.doc2bow(word) for word in lst_corpus]  ## train LDA
lda_model = gensim.models.ldamodel.LdaModel(corpus=dic_corpus, id2word=id2word, num_topics=3, random_state=123,update_every=1, chunksize=100, passes=10, alpha='auto',per_word_topics=True)

## output
lst_dics = []
for i in range(0, 3):
    lst_tuples = lda_model.get_topic_terms(i)
    for tupla in lst_tuples:
        lst_dics.append({"topic": i, "id": tupla[0],"word": id2word[tupla[0]],"weight": tupla[1]})
dtf_topics = pd.DataFrame(lst_dics,columns=['topic', 'id', 'word', 'weight'])

## plot
fig, ax = plt.subplots()
sns.barplot(y="word", x="weight", hue="topic", data=dtf_topics, dodge=False, ax=ax).set_title('Main Topics')
ax.set(ylabel="", xlabel="Word Importance")
plt.show()



dtf_topics = dtf.copy()

### Starting point
dtf = dtf_topics.copy()

'''

'''
TRAIN TEST SPLIT
'''
all_journals = dtf[["Source Title", label_field]].drop_duplicates().copy()
heterodox_test = all_journals["Source Title"][all_journals[label_field] == "heterodox"].sample(frac=0.3).copy()
samequality_test = all_journals["Source Title"][all_journals[label_field] == "samequality"].sample(frac=0.3).copy()
all_test = heterodox_test.append(samequality_test)

## split dataset
dtf_test = dtf[dtf["Source Title"].isin(all_test)]
dtf_train = dtf[~dtf["Source Title"].isin(all_test)]

dtf_train = dtf_train[[text_field_clean, label_field]]
dtf_test = dtf_test[[text_field_clean, label_field]]

#dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=0.3)


'''
BAG OF WORDS
'''
logger.info("1")
#balanbce dataset
from imblearn.under_sampling import RandomUnderSampler
under_sampler = RandomUnderSampler(random_state=42)
X_train, y_train = under_sampler.fit_resample(pd.DataFrame({"X": dtf_train[text_field_clean]}), pd.DataFrame({"y":dtf_train[label_field]}))

## get target
#y_train = y_train.values
#y_test = y_test.values
logger.info("2")
fig, ax = plt.subplots()
fig.suptitle(label_field, fontsize=12)
y_train.value_counts().plot(kind="barh", legend=False, ax=ax).grid(axis='x')
plt.show()


#FEATURE ENGINEERING
logger.info("3")
## Count (classic BoW)
vectorizer = feature_extraction.text.CountVectorizer(max_features=10000, ngram_range=(1,2))

## Tf-Idf (advanced variant of BoW)
vectorizer = feature_extraction.text.TfidfVectorizer(max_features=10000, ngram_range=(1,2))
logger.info("4")
corpus = X_train["X"]

vectorizer.fit(corpus)
X_train = vectorizer.transform(corpus)
dic_vocabulary = vectorizer.vocabulary_

if plot == True:
    sns.heatmap(X_train.todense()[:,np.random.randint(0,X_train.shape[1],100)]==0, vmin=0, vmax=1, cbar=False).set_title('Sparse Matrix Sample')

'''
word = "new york"

dic_vocabulary[word]
'''




#FEATURE SELECTION
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
vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)

vectorizer.fit(corpus)
X_train = vectorizer.transform(corpus)
dic_vocabulary = vectorizer.vocabulary_

#classify
classifier = naive_bayes.MultinomialNB()

## pipeline
model = pipeline.Pipeline([("vectorizer", vectorizer),
                           ("classifier", classifier)])

## train classifier
y_train = y_train.values.ravel()

model["classifier"].fit(X_train, y_train)

## test
X_test = dtf_test[text_field_clean].values
predicted = model.predict(X_test)
predicted_prob = model.predict_proba(X_test)

y_test = dtf_test[label_field].values
classes = np.unique(y_test)
y_test_array = pd.get_dummies(y_test, drop_first=False).values

## Accuracy, Precision, Recall
accuracy = metrics.accuracy_score(y_test, predicted)
auc = metrics.roc_auc_score(y_test, predicted_prob[:,1], multi_class="ovr")
print("Accuracy:", round(accuracy, 2))
print("Auc:", round(auc, 2))
print("Detail:")
print(metrics.classification_report(y_test, predicted))

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
i = 5 #5 is a good example
txt_instance = dtf_test["text_clean"].iloc[i]

## check true value and predicted value
print("True:", y_test[i], "--> Pred:", predicted[i], "| Prob:", round(np.max(predicted_prob[i]),2))

import IPython
## show explanation
explainer = lime_text.LimeTextExplainer(class_names=np.unique(y_train))
explained = explainer.explain_instance(txt_instance, model.predict_proba, num_features=9, top_labels=3)
explained.show_in_notebook(text=txt_instance, predict_proba=False)
explained.save_to_file('lime.html')



'''
WORD FREQUENCY
'''
print("WORD FREQUENCY")

if plot == True:
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