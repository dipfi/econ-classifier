import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

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


sns.set_style('whitegrid')
sns.set_context("paper")

results_folder = "results_final"
models = ["TFIDF", "W2V", "BERT"]
plots_tabels_folder = "plots_and_tables"

save_plots = False
save_tables = False

def load_data(information,
              model,
              results_folder):
    dtf = pd.read_csv(data_path + "/" + results_folder + "/" + information + "_" + model + ".csv")
    return dtf






#################
# MODEL SELECTION
#################


# TFIDF
#################

dtf_model_selection_tfidf = load_data(information = "1Model_Selection", model = "TFIDF", results_folder = results_folder)

'''
dtf_model_selection_tfidf_pivot = dtf_model_selection_tfidf.pivot(index = ["min_df","p_value_limit","ngram_range"],
                                                                    columns = "tfidf_classifier",
                                                                    values = ["AUC_PR"])

dtf_model_selection_tfidf_pivot.reset_index(level=['min_df', 'p_value_limit', "ngram_range"], inplace=True)
'''

if save_plots:
    fig, ax = plt.subplots(figsize= (10,6))
    sns.lineplot(data = dtf_model_selection_tfidf[["tfidf_classifier","min_df", "AUC_PR", "p_value_limit"]][dtf_model_selection_tfidf["ngram_range"] == "(1, 1)"],
                 x = "p_value_limit",
                 y = "AUC_PR",
                 hue = "tfidf_classifier",
                 style = "min_df")
    ax.set_title("AUC-PR Score for term-frequency based classification systems", fontsize = 15)
    plt.savefig(data_path + "/" + plots_tabels_folder + "/model_selection_tfidf_auc_pr")


    fig, ax = plt.subplots(figsize= (10,6))
    sns.lineplot(data = dtf_model_selection_tfidf[["number_relevant_features","min_df", "AUC_PR", "p_value_limit", "ngram_range"]],
                 x = "p_value_limit",
                 y = "number_relevant_features",
                 hue = "min_df",
                 style = "ngram_range").set(yscale = "log")
    ax.set_title("Number of features for term-frequency based classification systems", fontsize = 15)
    plt.savefig(data_path + "/" + plots_tabels_folder + "/model_selection_tfidf_num_features")

'''
plot_duration = sns.lineplot(data = dtf_model_selection_tfidf[["number_relevant_features","duration", "tfidf_classifier", "ngram_range"]], x = "number_relevant_features", y = "duration", hue = "tfidf_classifier", style = "ngram_range")
plot_duration.set(xscale = "log")
'''

dtf_model_selection_tfidf_latex = dtf_model_selection_tfidf[["tfidf_classifier", "min_df", "p_value_limit", "ngram_range", "AUC_PR", "number_relevant_features"]].sort_values(by=["tfidf_classifier", "min_df", "p_value_limit", "ngram_range"])

if save_tables:
    with open(data_path + "/" + plots_tabels_folder + "/model_selection_tfidf_table.tex",'w') as tf:
        tf.write(dtf_model_selection_tfidf_latex.to_latex(index = False))

# W2V
#################

dtf_model_selection_w2v = load_data(information = "1Model_Selection", model = "W2V", results_folder = results_folder)

if save_plots:
    fig, ax = plt.subplots(figsize= (10,6))
    sns.lineplot(data = dtf_model_selection_w2v[["num_epochs_for_embedding","window_size", "AUC_PR", "num_epochs_for_classification"]],
                 x = "window_size",
                 y = "AUC_PR",
                 hue = "num_epochs_for_embedding",
                 style = "num_epochs_for_classification")
    ax.set_title("AUC-PR Score for word-embeddings based classification system (word2vec + LSTM)", fontsize = 15)
    plt.savefig(data_path + "/" + plots_tabels_folder + "/model_selection_w2v_auc_score")

'''
plot_num_features = sns.lineplot(data = dtf_model_selection_w2v[["num_epochs_for_embedding","window_size", "duration", "num_epochs_for_classification"]], x = "window_size", y = "duration", hue = "num_epochs_for_embedding", style = "num_epochs_for_classification")
plot_num_features.set(xscale = "log")
'''

dtf_model_selection_w2v_latex = dtf_model_selection_w2v[["num_epochs_for_embedding","window_size", "num_epochs_for_classification", "AUC_PR"]].sort_values(by=["num_epochs_for_embedding","window_size", "num_epochs_for_classification"])

if save_tables:
    with open(data_path + "/" + plots_tabels_folder + "/model_selection_w2v_table.tex",'w') as tf:
        tf.write(dtf_model_selection_w2v_latex.to_latex(index = False))


# BERT
#################
dtf_model_selection_bert = load_data(information = "1Model_Selection", model = "BERT", results_folder = results_folder)

if save_plots:
    fig, ax = plt.subplots(figsize= (10,6))
    sns.lineplot(data = dtf_model_selection_bert[["bert_epochs","small_model", "AUC_PR"]],
                 x = "bert_epochs",
                 y = "AUC_PR",
                 hue = "small_model")
    ax.set_title("AUC-PR Score for classification system with transformers", fontsize = 15)
    plt.savefig(data_path + "/" + plots_tabels_folder + "/model_selection_bert_auc_score")

'''
plot_num_features = sns.lineplot(data = dtf_model_selection_bert[["bert_epochs","small_model", "duration"]], x = "bert_epochs", y = "duration", hue = "small_model")
plot_num_features.set(xscale = "log")
'''

dtf_model_selection_bert_latex = dtf_model_selection_bert[["small_model", "bert_epochs", "AUC_PR"]].sort_values(by=["small_model", "bert_epochs"])

if save_tables:
    with open(data_path + "/" + plots_tabels_folder + "/model_selection_bert_table.tex",'w') as tf:
        tf.write(dtf_model_selection_bert_latex.to_latex(index = False))








##################################
# PERFORMANCE JOURNAL BY JOURNAL
##################################

# AGGREGATE
###########

dtf_journals = pd.DataFrame()
for model in models:
    dtf_journals = pd.concat([dtf_journals, load_data(information = "4Journals", model = model, results_folder = results_folder)])

dtf_journals = dtf_journals[["test_journal","current_model","Label","Support_Negative","Support_Positive","Recall", "AVG_PRED_PROB"]]

dtf_journals_pivot = dtf_journals.pivot(index=["Label","test_journal"], columns="current_model", values = ["Recall","AVG_PRED_PROB"])
dtf_journals_pivot.reset_index(level = ["Label","test_journal"], inplace = True)


# Full Tables
###########

dtf_journals_pivot_recall = dtf_journals_pivot[["Label", "test_journal", "Recall"]]
dtf_journals_pivot_recall.columns = dtf_journals_pivot_recall.columns.to_flat_index()
dtf_journals_pivot_recall.columns = ["Label","Journal","Recall Bert","Recall TFIDF","Recall W2V"]
dtf_journals_pivot_recall["Journal"] = dtf_journals_pivot_recall["Journal"].str.slice(0,25).copy()

if save_tables:
    with open(data_path + "/" + plots_tabels_folder + "/journals_table_recall.tex",'w') as tf:
        tf.write(dtf_journals_pivot_recall.to_latex(index = False))


dtf_journals_pivot_AVG_PRED_PROB = dtf_journals_pivot[["Label", "test_journal", "AVG_PRED_PROB"]]
dtf_journals_pivot_AVG_PRED_PROB.columns = dtf_journals_pivot_AVG_PRED_PROB.columns.to_flat_index()
dtf_journals_pivot_AVG_PRED_PROB.columns = ["Label","Journal","AVG_PRED_PROB Bert","AVG_PRED_PROB TFIDF","AVG_PRED_PROB W2V"]
dtf_journals_pivot_AVG_PRED_PROB["Journal"] = dtf_journals_pivot_AVG_PRED_PROB["Journal"].str.slice(0,25).copy()

if save_tables:
    with open(data_path + "/" + plots_tabels_folder + "/journals_table_AVG_PRED_PROB.tex",'w') as tf:
        tf.write(dtf_journals_pivot_AVG_PRED_PROB.to_latex(index = False))





# Boxplots
###########

dtf_journals_pivot.groupby("Label").mean(["Recall","AVG_PRED_PROB"])

if save_plots:
    fig, ax = plt.subplots(figsize= (10,6))
    sns.boxplot(data = dtf_journals,
                x = "current_model",
                y = "Recall",
                hue="Label")
    ax.set_title("Average Recall by Journal", fontsize = 15)
    plt.savefig(data_path + "/" + plots_tabels_folder + "/journals_plot_recall")

if save_plots:
    fig, ax = plt.subplots(figsize= (10,6))
    sns.boxplot(data = dtf_journals,
                x = "current_model",
                y = "AVG_PRED_PROB",
                hue="Label")
    ax.set_title("Average Prediction Probability by Journal", fontsize = 15)
    plt.savefig(data_path + "/" + plots_tabels_folder + "/journals_plot_AVG_PRED_PROB")



# Correlation Tables
###########

dtf_journals_recall_correlation = dtf_journals.pivot(index="test_journal", columns="current_model", values = "Recall").corr()
dtf_journals_orthodox_recall_correlation = dtf_journals[dtf_journals["Label"]=="0orthodox"].pivot(index="test_journal", columns="current_model", values = "Recall").corr()
dtf_journals_heterodox_recall_correlation = dtf_journals[dtf_journals["Label"]=="1heterodox"].pivot(index="test_journal", columns="current_model", values = "Recall").corr()

dtf_journals_AVG_PRED_PROB_correlation = dtf_journals.pivot(index="test_journal", columns="current_model", values = "AVG_PRED_PROB").corr()
dtf_journals_orthodox_AVG_PRED_PROB_correlation = dtf_journals[dtf_journals["Label"]=="0orthodox"].pivot(index="test_journal", columns="current_model", values = "AVG_PRED_PROB").corr()
dtf_journals_heterodox_AVG_PRED_PROB_correlation = dtf_journals[dtf_journals["Label"]=="1heterodox"].pivot(index="test_journal", columns="current_model", values = "AVG_PRED_PROB").corr()

if save_tables:
    with open(data_path + "/" + plots_tabels_folder + "/journals_orthodox_table_recall_corr.tex",'w') as tf:
        tf.write(dtf_journals_orthodox_recall_correlation.to_latex(index = False))

    with open(data_path + "/" + plots_tabels_folder + "/journals_heterdox_table_recall_corr.tex",'w') as tf:
        tf.write(dtf_journals_heterodox_recall_correlation.to_latex(index = False))

    with open(data_path + "/" + plots_tabels_folder + "/journals_orthodox_table_avg_pred_prob_corr.tex", 'w') as tf:
        tf.write(dtf_journals_orthodox_AVG_PRED_PROB_correlation.to_latex(index=False))

    with open(data_path + "/" + plots_tabels_folder + "/journals_heterodox_table_avg_pred_prob_corr.tex", 'w') as tf:
        tf.write(dtf_journals_heterodox_AVG_PRED_PROB_correlation.to_latex(index=False))













####################
# TOP 5
###################

dtf_top5 = pd.DataFrame()
for model in models:
    dtf_temp = load_data(information="5Top5", model=model, results_folder=results_folder)
    dtf_temp["model"] = model
    dtf_temp["rank"] = dtf_temp["predicted_prob"].rank(ascending = False)
    dtf_top5 = pd.concat([dtf_top5, dtf_temp])



# aggregate
###########

dtf_top5_pivot = dtf_top5.pivot(index=["journal","pubyear","author","times_cited","title","abstract", "keywords_author", "keywords_plus", "WOS_ID"],
                                columns = "model",
                                values = ["predicted", "predicted_bin", "predicted_prob", "rank"])

dtf_top5_pivot.reset_index(level=["journal","pubyear","author","times_cited","title","abstract", "keywords_author", "keywords_plus", "WOS_ID"], inplace=True)

# Correlation Tables
###################

dtf_top5_predicted_prob_correlation = dtf_top5_pivot["predicted_prob"].astype(float).corr()
dtf_top5_rank_correlation = dtf_top5_pivot["rank"].astype(float).corr()

if save_tables:
    with open(data_path + "/" + plots_tabels_folder + "/top5_table_predicted_prob_corr.tex",'w') as tf:
        tf.write(dtf_top5_predicted_prob_correlation.to_latex(index = False))

    with open(data_path + "/" + plots_tabels_folder + "/top5_table_rank_corr.tex",'w') as tf:
        tf.write(dtf_top5_rank_correlation.to_latex(index = False))




dtf_top5_pivot.columns = dtf_top5_pivot.columns.to_flat_index()

pd.crosstab(dtf_top5_pivot[("predicted","TFIDF")],dtf_top5_pivot[("predicted","W2V")])

pd.crosstab(dtf_top5_pivot[("predicted","TFIDF")],dtf_top5_pivot[("predicted","BERT")])

pd.crosstab(dtf_top5_pivot[("predicted","W2V")],dtf_top5_pivot[("predicted","BERT")])



dtf_top5_tfidf_heterodox40 = dtf_top5_pivot.sort_values(by = ("rank","TFIDF"))[0:40]

dtf_top5_tfidf_orthodox40 = dtf_top5_pivot.sort_values(by = ("rank","TFIDF"), ascending=False)[0:40]

if save_tables:
    with open(data_path + "/" + plots_tabels_folder + "/top5_table_top_40_most_heterodox.tex",'w') as tf:
        tf.write(dtf_top5_tfidf_heterodox40.to_latex(index = False))

    with open(data_path + "/" + plots_tabels_folder + "/top5_table_top_40_most_orthodox.tex",'w') as tf:
        tf.write(dtf_top5_tfidf_orthodox40.to_latex(index = False))

bins = np.arange(1990, 2025, 5).tolist()
dtf_top5['pubyear_bin'] = pd.cut(dtf_top5['pubyear'], bins).astype("string")

dtf_top5_tfidf_agg = dtf_top5[dtf_top5["model"]=="TFIDF"].groupby(["journal", "pubyear_bin"]).mean(["predicted_bin","predicted_prob","rank"])
dtf_top5_tfidf_agg.reset_index(level=["journal", "pubyear_bin"], inplace = True)



sns.lineplot(data = dtf_top5_tfidf_agg[["pubyear_bin","journal", "predicted_prob"]], x = "pubyear_bin", y = "predicted_prob", hue = "journal")



if save_plots:
    fig, ax = plt.subplots(figsize= (10,6))
    sns.lineplot(data = dtf_top5_tfidf_agg[["pubyear_bin", "journal", "predicted_bin"]],
                 x = "pubyear_bin",
                 y = "predicted_bin",
                 hue = "journal")
    ax.set_title("Average heterodoxy score by journal over time", fontsize = 15)
    plt.savefig(data_path + "/" + plots_tabels_folder + "/top5_plot_heterodoxy_score_timeline")


    fig, ax = plt.subplots(figsize= (10,6))
    sns.lineplot(data = dtf_top5_tfidf_agg[["pubyear_bin","journal", "predicted_prob"]],
                 x = "pubyear_bin",
                 y = "predicted_prob",
                 hue = "journal")

    ax.set_title("Average proportion of heterodox articles by journal over time", fontsize = 15)
    plt.savefig(data_path + "/" + plots_tabels_folder + "/top5_plot_heterodox_articles_timeline")




# authors
#####################

authors = [i.replace(" ","") for s in [i.split(";") for i in dtf_top5["author"]] for i in s]

authors_unique = list(set(authors))

authors_list = [i.split(",") for i in authors_unique]

sum([len(i)==1 for i in authors_list])
sum([len(i)==2 for i in authors_list])
sum([len(i)==3 for i in authors_list])

authors_list_clean = [i for i in authors_list if len(i)==2]
authors_clean = [i[0] + "," + i[1] for i in authors_list_clean]
authors_clean = [i.lower() for i in authors_clean]

authors_final = [i.lower() for i in [i[0] + "," + i[1][0] for i in authors_list_clean]]

authors = [i.replace(" ","").lower() for i in dtf_top5_pivot[('author', '')]]

map = dict(zip(authors_clean, authors_final))

for old, new in zip(authors_clean, authors_final):
    #print(old + "-->" + new)
    authors = [i.replace(old, new) for i in authors]


dtf_top5_pivot["author"] = authors
dtf_top5_pivot["author_list"] = [i.split(";") for i in authors]

dtf_top5_pivot_long = dtf_top5_pivot.explode("author_list", ignore_index=True)
dtf_top5_pivot_long["ind_author"] = [i for s in [i.split(";") for i in authors] for i in s]
dtf_top5_pivot_long.sort_values("ind_author", inplace = True)

numcols= [("predicted_prob", "TFIDF"),
          ("predicted_prob", "W2V"),
          ("predicted_prob", "BERT"),
          ("predicted_bin", "TFIDF"),
          ("predicted_bin", "W2V"),
          ("predicted_bin", "BERT"),
          ("rank", "TFIDF"),
          ("rank", "W2V"),
          ("rank", "BERT"),
          ("times_cited","")]

for i in numcols:
    dtf_top5_pivot_long[i] = pd.to_numeric(dtf_top5_pivot_long[i], errors='coerce')

dtf_top5_pivot_long[("author_list","")] = dtf_top5_pivot_long["author_list"].astype(object)

d = {"predicted_prob":"mean_predicted_prob", "predicted_bin":"sum_predicted_bin"}
dtf_top5_author_ranks = dtf_top5_pivot_long[["ind_author",("predicted_prob","TFIDF"),("times_cited",""),("WOS_ID","")]]
dtf_top5_author_ranks = dtf_top5_author_ranks.groupby("ind_author").agg({("predicted_prob","TFIDF"):'mean',
                                                                           ("times_cited",""):'sum',
                                                                           ("WOS_ID",""):'count'}).sort_values(("predicted_prob","TFIDF"), ascending=False)

dtf_top5_author_citations = dtf_top5_pivot_long[["ind_author",("predicted","TFIDF"),("times_cited",""),("WOS_ID","")]]
dtf_top5_author_citations = dtf_top5_author_citations.groupby(["ind_author", ("predicted","TFIDF")]).agg({("times_cited",""):'sum',
                                                                                                         ("WOS_ID",""):'count'})
dtf_top5_author_citations.reset_index(level=["ind_author",("predicted","TFIDF")], inplace=True)
dtf_top5_author_citations= dtf_top5_author_citations.pivot(index="ind_author",
                                                            columns=[("predicted","TFIDF")],
                                                            values=[("times_cited",""), ("WOS_ID","")])

dtf_top5_authors = dtf_top5_author_ranks.join(dtf_top5_author_citations)

dtf_top5_authors.columns = ["predicted_prob_tfidf","total_times_cited","total_publications","orthodox_citations","heterodox_citations","orthodox_publications","heterodox_publications"]


# citations
################

dtf_top5_authors["orthodox_rel_citations"] = dtf_top5_authors["orthodox_citations"]/dtf_top5_authors["orthodox_publications"]
dtf_top5_authors["heterodox_rel_citations"] = dtf_top5_authors["heterodox_citations"]/dtf_top5_authors["heterodox_publications"]
dtf_top5_authors["heterodox_citation_index"] = dtf_top5_authors["heterodox_rel_citations"]/dtf_top5_authors["orthodox_rel_citations"]
dtf_top5_authors["weighted_prop_of_het_citations"] = dtf_top5_authors["heterodox_rel_citations"]/(dtf_top5_authors["heterodox_rel_citations"]+dtf_top5_authors["orthodox_rel_citations"])

dtf_top5_authors_atleast5 = dtf_top5_authors[dtf_top5_authors["total_publications"] > 4]

dtf_top5_authors_atleast5_onehet = dtf_top5_authors_atleast5[dtf_top5_authors_atleast5["heterodox_citations"]>0]

dtf_top5_authors_atleast5_onehet["heterodox_citation_index"].median()
dtf_top5_authors_atleast5_onehet["weighted_prop_of_het_citations"] .mean()
st.ttest_1samp(dtf_top5_authors_atleast5_onehet["weighted_prop_of_het_citations"], popmean = 0.5)

st.t.interval(alpha=0.99,
              df=len(dtf_top5_authors_atleast5_onehet["weighted_prop_of_het_citations"])-1,
              loc=np.mean(dtf_top5_authors_atleast5_onehet["weighted_prop_of_het_citations"]),
              scale=st.sem(dtf_top5_authors_atleast5_onehet["weighted_prop_of_het_citations"]))


if save_plots:
    fig, ax = plt.subplots(figsize= (10,6))
    sns.histplot(data = dtf_top5_authors_atleast5_onehet["weighted_prop_of_het_citations"],
                 kde=True)
    ax.set_title("Weighted proportion of citations for heterodox journals", fontsize = 15)
    plt.axvline(np.mean(dtf_top5_authors_atleast5_onehet["weighted_prop_of_het_citations"]), color = "r")
    plt.savefig(data_path + "/" + plots_tabels_folder + "/top5_plot_weighted_prop_of_het_citations")



# heterodox publications
################

dtf_top5_authors_atleast5_onehet["prop_of_heterodox_publications"] = dtf_top5_authors_atleast5_onehet["heterodox_publications"]/dtf_top5_authors_atleast5_onehet["total_publications"]
dtf_top5_authors_mosthet_count = dtf_top5_authors_atleast5_onehet[["total_publications", "heterodox_publications", "prop_of_heterodox_publications"]].sort_values(["heterodox_publications", "prop_of_heterodox_publications"], ascending = False)[0:40]
dtf_top5_authors_mosthet_prop = dtf_top5_authors_atleast5_onehet[["total_publications", "heterodox_publications", "prop_of_heterodox_publications"]].sort_values(["prop_of_heterodox_publications", "heterodox_publications"], ascending = False)[0:40]

if save_tables:
    with open(data_path + "/" + plots_tabels_folder + "/top5_table_most_heterodox_count.tex",'w') as tf:
        tf.write(dtf_top5_authors_mosthet_count.to_latex(index = False))

    with open(data_path + "/" + plots_tabels_folder + "/top5_table_most_heterodox_proportion.tex",'w') as tf:
        tf.write(dtf_top5_authors_mosthet_prop.to_latex(index = False))






'''

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

'''
