'''
This is a reproduction of:

https://towardsdatascience.com/text-analysis-feature-engineering-with-nlp-502d6ea9225d
'''

## for data
import pandas as pd
import collections
import json

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import wordcloud

## for text processing
import re
import nltk

## for language detection
import langdetect ## for sentiment
from textblob import TextBlob## for ner
import spacy

## for vectorizer
from sklearn import feature_extraction, manifold

## for word embedding
import gensim.downloader as gensim_api

## for topic modeling
import gensim

import pandas as pd

from tqdm import tqdm
tqdm.pandas()

##Set up
import random
random.seed(10)
import configparser
config = configparser.ConfigParser()
import os
config.read(os.getcwd()+'/code/config.ini')
data_path=config['PATH']['data_path']
code_path=config['PATH']['code_path']
project_path=config['PATH']['project']


'''
INPUT PARAMETERS HERE
'''
'''
############################################
file_name = "sample_for_damian_lang_10000.csv"
data_orig = pd.read_csv(data_path + "/" + file_name)
############################################
'''

'''
READ DATA
'''

print("READ DATA")
json_path = data_path + '/News_Category_Dataset_v2.json'

lst_dics = []
with open(json_path, mode='r', errors='ignore') as json_file:
    for dic in json_file:
        lst_dics.append( json.loads(dic) )

## print the first one
lst_dics[0]
'''
'''
data_orig = data_orig.loc[~data_orig["language_short"].isin(["none","multiple"]),["abstract", "discipline"]]
data_orig.rename(columns = {"abstract":"text", "discipline":"y"}, inplace = True)
'''
'''
SUBSET DATA
'''
print("SUBSET DATA")
## create dtf with news-data
dtf = pd.DataFrame(lst_dics)

## filter categories
dtf = dtf[dtf["category"].isin(['ENTERTAINMENT','POLITICS','TECH']) ][["category","headline"]]

## rename columns
dtf = dtf.rename(columns={"category":"y", "headline":"text"})

dtf = dtf.sample(frac = 0.1)

# t 5 random rows
print(dtf.sample(5))

'''
x = "y"

fig, ax = plt.subplots()
fig.suptitle(x, fontsize=12)
dtf[x].reset_index().groupby(x).count().sort_values(by=
       "index").plot(kind="barh", legend=False,
        ax=ax).grid(axis='x')
plt.show()
'''


'''
LANGUAGE DETECTION
'''
print("LANGUAGE DETECTION")
'''
txt = dtf["text"].iloc[0]

print(txt, " --> ", langdetect.detect(txt))
'''

dtf['lang'] = dtf["text"].process_apply(lambda x: langdetect.detect(x) if x.strip() != "" else "")

print(dtf.head())

'''
x = "lang"

fig, ax = plt.subplots()
fig.suptitle(x, fontsize=12)
dtf[x].reset_index().groupby(x).count().sort_values(by=
       "index").plot(kind="barh", legend=False,
        ax=ax).grid(axis='x')
plt.show()
'''

dtf = dtf[dtf["lang"]=="en"]


'''
TEXT PREPROCESSING
'''
print("TEXT PREPROCESSING")
'''
print("--- original ---")
print(txt)

print("--- cleaning ---")
txt = re.sub(r'[^\w\s]', '', str(txt).lower().strip())
print(txt)

print("--- tokenization ---")
txt = txt.split()
print(txt)
'''

lst_stopwords = nltk.corpus.stopwords.words("english")
#print(lst_stopwords)

'''
print("--- remove stopwords ---")
txt = [word for word in txt if word not in lst_stopwords]
print(txt)

print("--- stemming ---")
ps = nltk.stem.porter.PorterStemmer()
print([ps.stem(word) for word in txt])

print("--- lemmatisation ---")
lem = nltk.stem.wordnet.WordNetLemmatizer()
print([lem.lemmatize(word) for word in txt])
'''

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


def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

    ## Tokenize (convert from string to list)
    lst_text = text.split()

    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in
                    lst_stopwords]

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

dtf["text_clean"] = dtf["text"].process_apply(lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, lst_stopwords=lst_stopwords))


print(dtf.head())
'''
print(dtf["text"].iloc[0], " --> ", dtf["text_clean"].iloc[0])
'''



'''
LENGTH ANALYSIS
'''
print("TLENGTH ANALYSIS")

dtf['word_count'] = dtf["text"].process_apply(lambda x: len(str(x).split(" ")))

dtf['char_count'] = dtf["text"].process_apply(lambda x: sum(len(word) for word in str(x).split(" ")))

dtf['sentence_count'] = dtf["text"].process_apply(lambda x: len(str(x).split(".")))

dtf['avg_word_length'] = dtf['char_count'] / dtf['word_count']

dtf['avg_sentence_lenght'] = dtf['word_count'] / dtf['sentence_count']

print(dtf.head())


'''
x, y = "char_count", "y"

fig, ax = plt.subplots(nrows=1, ncols=2)
fig.suptitle(x, fontsize=12)
for i in dtf[y].unique():
    sns.distplot(dtf[dtf[y]==i][x], hist=True, kde=False,
                 bins=10, hist_kws={"alpha":0.8},
                 axlabel="histogram", ax=ax[0])
    sns.distplot(dtf[dtf[y]==i][x], hist=False, kde=True,
                 kde_kws={"shade":True}, axlabel="density",
                 ax=ax[1])
ax[0].grid(True)
ax[0].legend(dtf[y].unique())
ax[1].grid(True)
plt.show()
'''





'''
SENTIMENT ANALYSIS
'''
print("SENTIMENT ANALYSIS")

dtf["sentiment"] = dtf["text_clean"].process_apply(lambda x: TextBlob(x).sentiment.polarity)
print(dtf.head())

print(dtf["text"].iloc[0], " --> ", dtf["sentiment"].iloc[0])


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