import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_ml import ConfusionMatrix

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

data_path, scripts_path, project_path = config()


results_folder = "results_final"
models = ["TFIDF", "W2V", "BERT"]


def load_data(information,
              model,
              results_folder):
    dtf = pd.read_csv(data_path + "/" + results_folder + "/" + information + "_" + model + ".csv")
    return dtf

dtf_model_selection_tfidf = load_data(information = "1Model_Selection", model = "TFIDF", results_folder = results_folder)

'''
dtf_model_selection_tfidf_pivot = dtf_model_selection_tfidf.pivot(index = ["min_df","p_value_limit","ngram_range"],
                                                                    columns = "tfidf_classifier",
                                                                    values = ["AUC_PR"])

dtf_model_selection_tfidf_pivot.reset_index(level=['min_df', 'p_value_limit', "ngram_range"], inplace=True)
'''

sns.lineplot(data = dtf_model_selection_tfidf[["tfidf_classifier","min_df", "AUC_PR", "p_value_limit"]][dtf_model_selection_tfidf["ngram_range"] == "(1, 1)"], x = "p_value_limit", y = "AUC_PR", hue = "tfidf_classifier", style = "min_df")


plot_num_features = sns.lineplot(data = dtf_model_selection_tfidf[["number_relevant_features","min_df", "AUC_PR", "p_value_limit", "ngram_range"]], x = "p_value_limit", y = "number_relevant_features", hue = "min_df", style = "ngram_range")
plot_num_features.set(yscale = "log")

plot_num_features = sns.lineplot(data = dtf_model_selection_tfidf[["number_relevant_features","duration", "tfidf_classifier", "ngram_range"]], x = "number_relevant_features", y = "duration", hue = "tfidf_classifier", style = "ngram_range")
plot_num_features.set(xscale = "log")




dtf_model_selection_w2v = load_data(information = "1Model_Selection", model = "W2V", results_folder = results_folder)

sns.lineplot(data = dtf_model_selection_w2v[["num_epochs_for_embedding","window_size", "AUC_PR", "num_epochs_for_classification"]], x = "window_size", y = "AUC_PR", hue = "num_epochs_for_embedding", style = "num_epochs_for_classification")

'''
plot_num_features = sns.lineplot(data = dtf_model_selection_w2v[["num_epochs_for_embedding","window_size", "duration", "num_epochs_for_classification"]], x = "window_size", y = "duration", hue = "num_epochs_for_embedding", style = "num_epochs_for_classification")
plot_num_features.set(xscale = "log")
'''


dtf_model_selection_bert = load_data(information = "1Model_Selection", model = "BERT", results_folder = results_folder)

sns.lineplot(data = dtf_model_selection_bert[["bert_epochs","small_model", "AUC_PR"]], x = "bert_epochs", y = "AUC_PR", hue = "small_model")

'''
plot_num_features = sns.lineplot(data = dtf_model_selection_bert[["bert_epochs","small_model", "duration"]], x = "bert_epochs", y = "duration", hue = "small_model")
plot_num_features.set(xscale = "log")
'''










dtf_journals = pd.DataFrame()
for model in models:
    dtf_journals = pd.concat([dtf_journals, load_data(information = "4Journals", model = model, results_folder = results_folder)])

dtf_journals = dtf_journals[["test_journal","current_model","Label","Support_Negative","Support_Positive","Recall", "AVG_PRED_PROB"]]

dtf_journals_pivot = dtf_journals.pivot(index=["Label","test_journal"], columns="current_model", values = ["Recall","AVG_PRED_PROB"])
dtf_journals_pivot.reset_index(level = ["Label","test_journal"], inplace = True)

dtf_journals_pivot.groupby("Label").mean(["Recall","AVG_PRED_PROB"])

sns.boxplot(data = dtf_journals, x = "current_model", y = "Recall", hue="Label")

sns.boxplot(data = dtf_journals, x = "current_model", y = "AVG_PRED_PROB", hue="Label")

dtf_journals.pivot(index="test_journal", columns="current_model", values = "Recall").corr()
dtf_journals[dtf_journals["Label"]=="0orthodox"].pivot(index="test_journal", columns="current_model", values = "Recall").corr()
dtf_journals[dtf_journals["Label"]=="1heterodox"].pivot(index="test_journal", columns="current_model", values = "Recall").corr()

dtf_journals.pivot(index="test_journal", columns="current_model", values = "AVG_PRED_PROB").corr()
dtf_journals[dtf_journals["Label"]=="0orthodox"].pivot(index="test_journal", columns="current_model", values = "AVG_PRED_PROB").corr()
dtf_journals[dtf_journals["Label"]=="1heterodox"].pivot(index="test_journal", columns="current_model", values = "AVG_PRED_PROB").corr()


dtf_top5 = pd.DataFrame()
for model in models:
    dtf_temp = load_data(information="5Top5", model=model, results_folder=results_folder)
    dtf_temp["model"] = model
    dtf_temp["rank"] = dtf_temp["predicted_prob"].rank(ascending = False)
    dtf_top5 = pd.concat([dtf_top5, dtf_temp])



dtf_top5_pivot = dtf_top5.pivot(index=["journal","pubyear","author","times_cited","title","abstract", "keywords_author", "keywords_plus", "WOS_ID"],
                                columns = "model",
                                values = ["predicted", "predicted_bin", "predicted_prob", "rank"])

dtf_top5_pivot.reset_index(level=["journal","pubyear","author","times_cited","title","abstract", "keywords_author", "keywords_plus", "WOS_ID"], inplace=True)


dtf_top5_pivot["predicted_prob"].astype(float).corr()
dtf_top5_pivot["rank"].astype(float).corr()

pd.crosstab(dtf_top5_pivot["predicted"]["TFIDF"],dtf_top5_pivot["predicted"]["W2V"])

pd.crosstab(dtf_top5_pivot["predicted"]["TFIDF"],dtf_top5_pivot["predicted"]["BERT"])

pd.crosstab(dtf_top5_pivot["predicted"]["W2V"],dtf_top5_pivot["predicted"]["BERT"])



dtf_top5_tfidf_heterodox40 = dtf_top5_pivot[dtf_top5_pivot["rank"]["TFIDF"]<41].sort_values(by = ("rank","TFIDF"))

dtf_top5_tfidf_orthodox40 = dtf_top5_pivot[dtf_top5_pivot["rank"]["TFIDF"]>len(dtf_top5_pivot)-41].sort_values(by = ("rank","TFIDF"), ascending=False)

dtf_top5_tfidf_agg = dtf_top5[dtf_top5["model"]=="TFIDF"].groupby(["journal", "pubyear"]).mean(["predicted_bin","predicted_prob","rank"])
dtf_top5_tfidf_agg.reset_index(level=["journal", "pubyear"], inplace = True)

sns.lineplot(data = dtf_top5_tfidf_agg[["pubyear","journal", "predicted_bin"]], x = "pubyear", y = "predicted_bin", hue = "journal")

sns.lineplot(data = dtf_top5_tfidf_agg[["pubyear","journal", "predicted_prob"]], x = "pubyear", y = "predicted_prob", hue = "journal")









authors = [i.replace(" ","") for s in [i.split(";") for i in dtf_top5["author"]] for i in s]

authors_unique = list(set(authors))

authors_list = [i.split(",") for i in authors_unique]

sum([len(i)==1 for i in authors_list])
sum([len(i)==2 for i in authors_list])
sum([len(i)==3 for i in authors_list])

authors_list_clean = [i for i in authors_list if len(i)==2]
authors_clean = [i[0] + "," + i[1] for i in authors_list_clean]

authors_final = [i.lower() for i in [i[0] + "," + i[1][0] for i in authors_list_clean]]

authors = [i.replace(" ","") for i in dtf_top5_pivot["author"]]

map = dict(zip(authors_clean, authors_final))

for old, new in zip(authors_clean, authors_final):
    #print(old + "-->" + new)
    authors = [i.replace(old, new) for i in authors]


dtf_top5_pivot["author"] = authors
dtf_top5_pivot["author_list"] = [i.split(";") for i in authors]

dtf_top5_pivot_long = dtf_top5_pivot.explode("author_list", ignore_index=True)
dtf_top5_pivot_long["ind_author"] = [i for s in [i.split(";") for i in authors] for i in s]


numcols= [("predicted_prob", "TFIDF"),
          ("predicted_prob", "W2V"),
          ("predicted_prob", "BERT"),
          ("predicted_bin", "TFIDF"),
          ("predicted_bin", "W2V"),
          ("predicted_bin", "BERT"),
          "times_cited"]

for i in numcols:
    dtf_top5_pivot_long[i] = dtf_top5_pivot_long[i].astype(float)

d = {"predicted_prob":"mean_predicted_prob", "predicted_bin":"sum_predicted_bin"}
dtf_top5_author_ranks = dtf_top5_pivot_long.groupby("ind_author").agg({("predicted_prob","TFIDF"):'mean',
                                                                   ("predicted_bin","TFIDF"):'sum'})



dtf_top5_pivot_long[("predicted_prob","TFIDF")]







dtf_top5_pivot_tfidf_orthodox = dtf_top5_pivot.loc[dtf_top5_pivot["predicted_bin"]["TFIDF"]==0]
dtf_top5_pivot_tfidf_heterodox = dtf_top5_pivot.loc[dtf_top5_pivot["predicted_bin"]["TFIDF"]==1]


publication_count = pd.DataFrame({"author": list(set(authors_final)),
                                  "orthodox_count": [0 for i in list(set(authors_final))],
                                  "heterodox_count": [0 for i in list(set(authors_final))],})

for i in list(set(authors_final)):
    orthodox_count = sum(1 for s in dtf_top5_pivot_tfidf_orthodox["author"] if i in s)
    heterodox_count = sum(1 for s in dtf_top5_pivot_tfidf_heterodox["author"] if i in s)
    publication_count.loc[publication_count["author"] == i, "orthodox_count"] = orthodox_count
    publication_count.loc[publication_count["author"] == i, "heterodox_count"] = heterodox_count

publication_count["total_count"] = publication_count["orthodox_count"] + publication_count["heterodox_count"]

publication_count["heterodox_proportion"] = publication_count["heterodox_count"] / publication_count["total_count"]

publication_count.sort_values("heterodox_count", ascending=False, inplace=True)

for i in list(set(authors_final)):




