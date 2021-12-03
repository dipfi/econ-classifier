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

##downloading wordlist in main
import nltk

'''
DISCIPLINES SPECIFIC IMPORTS
'''
## for data
import pandas as pd

pd.set_option('display.max_columns', None)



############################################
logging_level = logging.INFO  # logging.DEBUG #logging.WARNING
print_charts_tables = True  # False #True
input_file_name = "WOS_lee_heterodox_und_samequality_preprocessed"
input_file_size = 10000 #10000 #"all"
input_file_type = "csv"
output_file_name = "WOS_lee_heterodox_und_samequality_preprocessed_wip"
sample_size = 10000  #input_file_size #10000 #"all"
text_field = "Abstract"  # "title" #"abstract"
label_field = "labels"
cores = mp.cpu_count()  #mp.cpu_count()  #2
save = True  # False #True
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






'''
LENGTH ANALYSIS
'''

print("TLENGTH ANALYSIS")

dtf['word_count'] = dtf["text"].progress_apply(lambda x: len(str(x).split(" ")))

dtf['char_count'] = dtf["text"].progress_apply(lambda x: sum(len(word) for word in str(x).split(" ")))

dtf['sentence_count'] = dtf["text"].progress_apply(lambda x: len(str(x).split(".")))

dtf['avg_word_length'] = dtf['char_count'] / dtf['word_count']

dtf['avg_sentence_lenght'] = dtf['word_count'] / dtf['sentence_count']

dtf.groupby(by="y")[["avg_word_length","avg_sentence_lenght", "sentence_count", "word_count"]].mean()


import matplotlib.pyplot as plt
import seaborn as sns


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

## for sentiment
from textblob import TextBlob

print("SENTIMENT ANALYSIS")

dtf["polarity"] = dtf["text_clean"].progress_apply(lambda x: TextBlob(x).sentiment.polarity) #-1 = negative, +1=positive
dtf["subjectivity"] = dtf["text_clean"].progress_apply(lambda x: TextBlob(x).sentiment.subjectivity) #0 = objective, 1=subjective
dtf.groupby(by="y")[["polarity", "subjectivity"]].mean()

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




## call model
ner = spacy.load("en_core_web_lg")

## tag text
txt = dtf["text"].iloc[0]
doc = ner(txt)

## display result
spacy.displacy.render(doc, style="ent")




## tag text and exctract tags into a list
dtf["tags"] = dtf["text"].process_apply(lambda x: [(tag.text, tag.label_) for tag in ner(x).ents] )

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
dtf["tags"] = dtf["tags"].process_apply(lambda x: utils_lst_count(x))

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
     dtf["tags_"+feature] = dtf["tags"].process_apply(lambda x: utils_ner_features(x, feature))

## print result
print(dtf.head())


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


## predict wit NER
txt = dtf["text"].iloc[0]
entities = ner(txt).ents

## tag text
tagged_txt = txt
for tag in entities:
    tagged_txt = re.sub(tag.text, "_".join(tag.text.split()), tagged_txt)

## show result
print(tagged_txt)




'''
WORD FREQUENCY
'''
print("WORD FREQUENCY")
'''
y = "POLITICS"
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
dtf_bi["Word"] = dtf_bi["Word"].process_(lambda x: " ".join(string for string in x))
dtf_bi.set_index("Word").iloc[:top, :].sort_values(by="Freq").plot(kind="barh", title="Bigrams", ax=ax[1],legend=False).grid(axis='x')
ax[1].set(ylabel=None)
plt.show()
'''



lst_words = ["box office", "republican", "apple"]

## count
lst_grams = [len(word.split(" ")) for word in lst_words]
vectorizer = feature_extraction.text.CountVectorizer(
                 vocabulary=lst_words,
                 ngram_range=(min(lst_grams),max(lst_grams)))

dtf_X = pd.DataFrame(vectorizer.fit_transform(dtf["text_clean"]).todense(), columns=lst_words)

## add the new features as columns
dtf = pd.concat([dtf, dtf_X.set_index(dtf.index)], axis=1)
print(dtf.head())


'''
wc = wordcloud.WordCloud(background_color='black', max_words=100, max_font_size=35)
wc = wc.generate(str(corpus))
fig = plt.figure(num=1)
plt.axis('off')
plt.imshow(wc, cmap=None)
plt.show()
'''






'''
WORD VECTORS
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


'''
TOPIC MODELING
'''
print("TOPIC MODELING")

y = "TECH"
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