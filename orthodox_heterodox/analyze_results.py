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
plots_tables_folder = "plots_and_tables"

save_plots = True
save_tables = True


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

dtf_model_selection_tfidf = load_data(information="1Model_Selection", model="TFIDF", results_folder=results_folder)

'''
dtf_model_selection_tfidf_pivot = dtf_model_selection_tfidf.pivot(index = ["min_df","p_value_limit","ngram_range"],
                                                                    columns = "tfidf_classifier",
                                                                    values = ["AUC_PR"])

dtf_model_selection_tfidf_pivot.reset_index(level=['min_df', 'p_value_limit', "ngram_range"], inplace=True)
'''

if save_plots:

    dtf_model_selection_tfidf_latex = dtf_model_selection_tfidf[["number_relevant_features", "min_df", "AUC_PR", "p_value_limit", "ngram_range", "tfidf_classifier"]]
    dtf_model_selection_tfidf_latex.columns = ["Number of features", "Minimum document frequency", "AUC PR", "P-value threshold", "N-gram range", "Classifier"]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=dtf_model_selection_tfidf_latex[["Number of features", "Minimum document frequency", "AUC PR", "P-value threshold", "Classifier"]][dtf_model_selection_tfidf["ngram_range"] == "(1, 1)"],
                 x="P-value threshold",
                 y="AUC PR",
                 hue="Classifier",
                 style="Minimum document frequency")
    plt.xlabel("P-value threshold", fontsize=15)
    plt.ylabel("AUC-PR", fontsize=15)
    plt.savefig(data_path + "/" + plots_tables_folder + "/model_selection_tfidf_auc_pr")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=dtf_model_selection_tfidf_latex[["Number of features", "Minimum document frequency", "AUC PR", "P-value threshold", "N-gram range"]],
                 x="P-value threshold",
                 y="Number of features",
                 hue="Minimum document frequency",
                 style="N-gram range").set(yscale="log")
    plt.xlabel("P-value threshold", fontsize=15)
    plt.ylabel("Number of features", fontsize=15)
    plt.savefig(data_path + "/" + plots_tables_folder + "/model_selection_tfidf_num_features")
    plt.close()
'''
plot_duration = sns.lineplot(data = dtf_model_selection_tfidf[["number_relevant_features","duration", "tfidf_classifier", "ngram_range"]], x = "number_relevant_features", y = "duration", hue = "tfidf_classifier", style = "ngram_range")
plot_duration.set(xscale = "log")
'''

dtf_model_selection_tfidf_latex = dtf_model_selection_tfidf[["tfidf_classifier", "min_df", "p_value_limit", "ngram_range", "AUC_PR", "number_relevant_features"]].sort_values(by=["tfidf_classifier", "min_df", "p_value_limit", "ngram_range"])

if save_tables:
    with open(data_path + "/" + plots_tables_folder + "/model_selection_tfidf_table.tex", 'w') as tf:
        tf.write(dtf_model_selection_tfidf_latex.to_latex(caption="Model selection for classification systems based on term-frequencies",
                                                          label="model_selection_tfidf",
                                                          index=False))

# W2V
#################

dtf_model_selection_w2v = load_data(information="1Model_Selection", model="W2V", results_folder=results_folder)

dtf_model_selection_w2v_latex = dtf_model_selection_w2v[["num_epochs_for_embedding", "window_size", "AUC_PR", "num_epochs_for_classification"]]

dtf_model_selection_w2v_latex.columns = ["Number of epochs for embedding", "Window size for embedding", "AUC PR", "Number of epochs for classification"]

if save_plots:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=dtf_model_selection_w2v_latex,
                 x="Window size for embedding",
                 y="AUC PR",
                 hue="Number of epochs for embedding",
                 style="Number of epochs for classification")
    plt.xlabel("Window size for embedding", fontsize=15)
    plt.ylabel("AUC-PR", fontsize=15)
    plt.savefig(data_path + "/" + plots_tables_folder + "/model_selection_w2v_auc_score")
    plt.close()

'''
plot_num_features = sns.lineplot(data = dtf_model_selection_w2v[["num_epochs_for_embedding","window_size", "duration", "num_epochs_for_classification"]], x = "window_size", y = "duration", hue = "num_epochs_for_embedding", style = "num_epochs_for_classification")
plot_num_features.set(xscale = "log")
'''

'''
dtf_model_selection_w2v_latex = dtf_model_selection_w2v[["num_epochs_for_embedding", "window_size", "num_epochs_for_classification", "AUC_PR"]].sort_values(by=["num_epochs_for_embedding", "window_size", "num_epochs_for_classification"])

if save_tables:
    with open(data_path + "/" + plots_tables_folder + "/model_selection_w2v_table.tex", 'w') as tf:
        tf.write(dtf_model_selection_w2v_latex.to_latex(caption="Model selection for classification systems based on word embeddings",
                                                        label="model_selection_w2v",
                                                        index=False))
'''


# BERT
#################
dtf_model_selection_bert = load_data(information="1Model_Selection", model="BERT", results_folder=results_folder)

dtf_model_selection_bert["Model"] = ["DistilBert" if i else "Bert" for i in  dtf_model_selection_bert["small_model"]]
dtf_model_selection_bert_latex = dtf_model_selection_bert[["bert_epochs", "Model", "AUC_PR"]]

dtf_model_selection_bert_latex.columns = ["Training epochs", "Model", "AUC PR"]

if save_plots:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=dtf_model_selection_bert_latex,
                 x="Training epochs",
                 y="AUC PR",
                 hue="Model")
    plt.xlabel("Training Epochs", fontsize=15)
    plt.ylabel("AUC-PR", fontsize=15)
    plt.savefig(data_path + "/" + plots_tables_folder + "/model_selection_bert_auc_score")
    plt.close()

'''
plot_num_features = sns.lineplot(data = dtf_model_selection_bert[["bert_epochs","small_model", "duration"]], x = "bert_epochs", y = "duration", hue = "small_model")
plot_num_features.set(xscale = "log")
'''
'''
dtf_model_selection_bert_latex = dtf_model_selection_bert[["small_model", "bert_epochs", "AUC_PR"]].sort_values(by=["small_model", "bert_epochs"])

if save_tables:
    with open(data_path + "/" + plots_tables_folder + "/model_selection_bert_table.tex", 'w') as tf:
        tf.write(dtf_model_selection_bert_latex.to_latex(caption="Model selection for classification systems based on Transformers",
                                                         label="model_selection_bert",
                                                         index=False))
'''









##################################
# PERFORMANCE JOURNAL BY JOURNAL
##################################

# AGGREGATE
###########

dtf_journals = pd.DataFrame()
for model in models:
    dtf_journals = pd.concat([dtf_journals, load_data(information="4Journals", model=model, results_folder=results_folder)])

dtf_journals = dtf_journals[["test_journal", "current_model", "Label", "Support_Negative", "Support_Positive", "Recall", "AVG_PRED_PROB"]]

dtf_journals_pivot = dtf_journals.pivot(index=["Label", "test_journal"], columns="current_model", values=["Recall", "AVG_PRED_PROB"])
dtf_journals_pivot.reset_index(level=["Label", "test_journal"], inplace=True)





# Full Tables
###########

dtf_journals_pivot_recall = dtf_journals_pivot[["Label", "test_journal", "Recall"]]
dtf_journals_pivot_recall.columns = dtf_journals_pivot_recall.columns.to_flat_index()
dtf_journals_pivot_recall.columns = ["Label", "Journal", "Recall DISTILBERT", "Recall TFIDF", "Recall WORD2VEC"]
dtf_journals_pivot_recall = dtf_journals_pivot_recall.round(2)

if save_tables:
    with open(data_path + "/" + plots_tables_folder + "/journals_table_recall_orthodox.tex", 'w') as tf:
        tf.write(dtf_journals_pivot_recall[["Journal", "Recall DISTILBERT", "Recall TFIDF", "Recall WORD2VEC"]][dtf_journals_pivot_recall["Label"]=="0orthodox"].to_latex(index=False,
                                                                                                                                   escape = False,
                                                                                                                                   column_format="lccc"))
    with open(data_path + "/" + plots_tables_folder + "/journals_table_recall_heterodox.tex", 'w') as tf:
        tf.write(dtf_journals_pivot_recall[["Journal", "Recall DISTILBERT", "Recall TFIDF", "Recall WORD2VEC"]][dtf_journals_pivot_recall["Label"]=="1heterodox"].to_latex(index=False,
                                                                                                                                   escape = False,
                                                                                                                                   column_format="lccc"))

dtf_journals_pivot_AVG_PRED_PROB = dtf_journals_pivot[["Label", "test_journal", "AVG_PRED_PROB"]]
dtf_journals_pivot_AVG_PRED_PROB.columns = dtf_journals_pivot_AVG_PRED_PROB.columns.to_flat_index()
dtf_journals_pivot_AVG_PRED_PROB.columns = ["Label", "Journal", "Het. score DISTILBERT", "Het. score TFIDF", "Het. score WORD2VEC"]
dtf_journals_pivot_AVG_PRED_PROB = dtf_journals_pivot_AVG_PRED_PROB.round(2)

if save_tables:
    with open(data_path + "/" + plots_tables_folder + "/journals_table_hetscore_orthodox.tex", 'w') as tf:
        tf.write(dtf_journals_pivot_AVG_PRED_PROB[["Journal", "Het. score DISTILBERT", "Het. score TFIDF", "Het. score WORD2VEC"]][dtf_journals_pivot_AVG_PRED_PROB["Label"]=="0orthodox"].to_latex(index=False,
                                                                                                                                   escape = False,
                                                                                                                                   column_format="lccc"))
    with open(data_path + "/" + plots_tables_folder + "/journals_table_hetscore_heterodox.tex", 'w') as tf:
        tf.write(dtf_journals_pivot_AVG_PRED_PROB[["Journal", "Het. score DISTILBERT", "Het. score TFIDF", "Het. score WORD2VEC"]][dtf_journals_pivot_AVG_PRED_PROB["Label"]=="1heterodox"].to_latex(index=False,
                                                                                                                                   escape = False,
                                                                                                                                   column_format="lccc"))

# Boxplots
###########

dtf_journals_pivot.groupby("Label").mean(["Recall", "AVG_PRED_PROB"])

dtf_journals_latex = dtf_journals[["current_model", "Recall", "Label","AVG_PRED_PROB"]]
dtf_journals_latex["current_model"].replace({"bert":"DISTILBERT", "tfidf": "TFIDF", "w2v":"WORD2VEC"}, inplace = True)
dtf_journals_latex.rename(columns={"current_model":"Model", "AVG_PRED_PROB": "Heterodoxy score"}, inplace = True)
dtf_journals_latex.round(2)

if save_plots:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=dtf_journals_latex,
                x="Model",
                y="Recall",
                hue="Label")
    plt.xlabel("Model", fontsize=15)
    plt.ylabel("Recall", fontsize=15)
    plt.savefig(data_path + "/" + plots_tables_folder + "/journals_plot_recall")
    plt.close()

if save_plots:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=dtf_journals_latex,
                x="Model",
                y="Heterodoxy score",
                hue="Label")
    plt.xlabel("Model", fontsize=15)
    plt.ylabel("Heterodoxy score", fontsize=15)
    plt.savefig(data_path + "/" + plots_tables_folder + "/journals_plot_AVG_PRED_PROB")
    plt.close()









# Means
########
'''
dtf_journals_pivot_recall = dtf_journals[["test_journal", "current_model", "Label", "Recall"]].pivot(index=["test_journal", "current_model"],
                                                                                                     columns="Label",
                                                                                                     values="Recall")
dtf_journals_pivot_recall = dtf_journals_pivot_recall.reset_index(["test_journal", "current_model"])
dtf_journals_means_recall = dtf_journals_pivot_recall.groupby("current_model").mean("Recall")

dtf_journals_pivot_AVG_PRED_PROB = dtf_journals[["test_journal", "current_model", "Label", "AVG_PRED_PROB"]].pivot(index=["test_journal", "current_model"],
                                                                                                                   columns="Label",
                                                                                                                   values="AVG_PRED_PROB")
dtf_journals_pivot_AVG_PRED_PROB = dtf_journals_pivot_AVG_PRED_PROB.reset_index(["test_journal", "current_model"])
dtf_journals_means_AVG_PRED_PROB = dtf_journals_pivot_AVG_PRED_PROB.groupby("current_model").mean("AVG_PRED_PROB")


dtf_journals_means_recall_latex = dtf_journals_means_recall.reset_index()
dtf_journals_means_recall_latex["current_model"].replace({"bert":"DISTILBERT", "tfidf": "TFIDF", "w2v":"WORD2VEC"}, inplace = True)
dtf_journals_means_recall_latex = dtf_journals_means_recall_latex[["current_model", "0orthodox","1heterodox"]].round(2)
dtf_journals_means_recall_latex.columns = ["Current Model","Avg. recall (orthodox)","Avg. recall (heterodox)"]

if save_tables:
    with open(data_path + "/" + plots_tables_folder + "/journals_means_recall.tex", 'w') as tf:
        tf.write(dtf_journals_means_recall_latex.to_latex(index=False,
                                                   escape = False,
                                                   column_format="lcc"))


if save_tables:
    with open(data_path + "/" + plots_tables_folder + "/journals_means_AVG_PRED_PROB.tex", 'w') as tf:
        tf.write(dtf_journals_means_AVG_PRED_PROB.to_latex(caption="Average heterodoxy score by model and label",
                                                           label="journals_means_AVG_PRED_PROB"))



'''







# most heterodox/orthodox
#######################

# Correlation Tables
###########

dtf_journals_recall_correlation = dtf_journals.pivot(index="test_journal", columns="current_model", values="Recall").corr()
dtf_journals_orthodox_recall_correlation = dtf_journals[dtf_journals["Label"] == "0orthodox"].pivot(index="test_journal", columns="current_model", values="Recall").corr().round(2)
dtf_journals_heterodox_recall_correlation = dtf_journals[dtf_journals["Label"] == "1heterodox"].pivot(index="test_journal", columns="current_model", values="Recall").corr().round(2)

dtf_journals_orthodox_recall_correlation.reset_index(inplace = True)
dtf_journals_orthodox_recall_correlation.replace({"bert":"DISTILBERT", "tfidf":"TFIDF", "w2v":"WORD2VEC"}, inplace = True)
dtf_journals_orthodox_recall_correlation.rename(columns = {"bert":"DISTILBERT", "tfidf":"TFIDF", "w2v":"WORD2VEC"}, inplace = True)

dtf_journals_heterodox_recall_correlation.reset_index(inplace = True)
dtf_journals_heterodox_recall_correlation.replace({"bert":"DISTILBERT", "tfidf":"TFIDF", "w2v":"WORD2VEC"}, inplace = True)
dtf_journals_heterodox_recall_correlation.rename(columns = {"bert":"DISTILBERT", "tfidf":"TFIDF", "w2v":"WORD2VEC"}, inplace = True)

if save_tables:
    with open(data_path + "/" + plots_tables_folder + "/journals_orthodox_table_recall_corr.tex", 'w') as tf:
        tf.write(dtf_journals_orthodox_recall_correlation.to_latex(index=False,
                                                                   escape = False,
                                                                   column_format="lccc"))

    with open(data_path + "/" + plots_tables_folder + "/journals_heterodox_table_recall_corr.tex", 'w') as tf:
        tf.write(dtf_journals_heterodox_recall_correlation.to_latex(index=False,
                                                                   escape = False,
                                                                   column_format="lccc"))

'''
dtf_journals_AVG_PRED_PROB_correlation = dtf_journals.pivot(index="test_journal", columns="current_model", values="AVG_PRED_PROB").corr()
dtf_journals_orthodox_AVG_PRED_PROB_correlation = dtf_journals[dtf_journals["Label"] == "0orthodox"].pivot(index="test_journal", columns="current_model", values="AVG_PRED_PROB").corr()
dtf_journals_heterodox_AVG_PRED_PROB_correlation = dtf_journals[dtf_journals["Label"] == "1heterodox"].pivot(index="test_journal", columns="current_model", values="AVG_PRED_PROB").corr()


if save_tables:
    with open(data_path + "/" + plots_tables_folder + "/journals_orthodox_table_avg_pred_prob_corr.tex", 'w') as tf:
        tf.write(dtf_journals_orthodox_AVG_PRED_PROB_correlation.to_latex(caption="Correlation between models for heterodoxy score of orthodox journals",
                                                                          label="journals_orthodox_table_avg_pred_prob_corr",
                                                                          index=False))

    with open(data_path + "/" + plots_tables_folder + "/journals_heterodox_table_avg_pred_prob_corr.tex", 'w') as tf:
        tf.write(dtf_journals_heterodox_AVG_PRED_PROB_correlation.to_latex(caption="Correlation between models for heterodoxy score of heterodox journals",
                                                                           label="journals_heterodox_table_avg_pred_prob_corr",
                                                                           index=False))

'''













####################
# TOP 5
###################

dtf_top5 = pd.DataFrame()
for model in models:
    dtf_temp = load_data(information="5Top5", model=model, results_folder=results_folder)
    dtf_temp["model"] = model
    dtf_temp["rank"] = dtf_temp["predicted_prob"].rank(ascending=False)
    dtf_top5 = pd.concat([dtf_top5, dtf_temp])

# aggregate
###########

dtf_top5_pivot = dtf_top5.pivot(index=["journal", "pubyear", "author", "times_cited", "title", "abstract", "keywords_author", "keywords_plus", "WOS_ID"],
                                columns="model",
                                values=["predicted", "predicted_bin", "predicted_prob", "rank"])

dtf_top5_pivot.reset_index(level=["journal", "pubyear", "author", "times_cited", "title", "abstract", "keywords_author", "keywords_plus", "WOS_ID"], inplace=True)

# Correlation Tables
###################

dtf_top5_predicted_prob_correlation = dtf_top5_pivot["predicted_prob"].astype(float).corr().round(2).reset_index()
dtf_top5_predicted_prob_correlation["model"].replace({"W2W":"WORD2VEC", "BERT":"DISTILBERT"}, inplace = True)
dtf_top5_predicted_prob_correlation.columns = ["Model", "DISTILBERT", "TFIDF", "WORD2VEC"]

dtf_top5_rank_correlation = dtf_top5_pivot["rank"].astype(float).corr().round(2).reset_index()
dtf_top5_rank_correlation["model"].replace({"W2W":"WORD2VEC", "BERT":"DISTILBERT"}, inplace = True)
dtf_top5_rank_correlation.columns = ["Model", "DISTILBERT", "TFIDF", "WORD2VEC"]

if save_tables:
    with open(data_path + "/" + plots_tables_folder + "/top5_table_predicted_prob_corr.tex", 'w') as tf:
        tf.write(dtf_top5_predicted_prob_correlation.to_latex(index=False,
                                                               escape = False,
                                                               column_format="lccc"))

    with open(data_path + "/" + plots_tables_folder + "/top5_table_rank_corr.tex", 'w') as tf:
        tf.write(dtf_top5_rank_correlation.to_latex(index=False,
                                                       escape = False,
                                                       column_format="lccc"))

dtf_top5_pivot.columns = dtf_top5_pivot.columns.to_flat_index()

dtf_top5_table_cm_TFIDF_W2V = pd.crosstab(dtf_top5_pivot[("predicted", "TFIDF")], dtf_top5_pivot[("predicted", "W2V")]).reset_index()
dtf_top5_table_cm_TFIDF_W2V.columns = ["TFIDF/WORD2VEC", "0orthodox", "1heterodox"]


dtf_top5_table_cm_TFIDF_BERT = pd.crosstab(dtf_top5_pivot[("predicted", "TFIDF")], dtf_top5_pivot[("predicted", "BERT")]).reset_index()
dtf_top5_table_cm_TFIDF_BERT.columns = ["TFIDF/DISTILBERT", "0orthodox", "1heterodox"]

dtf_top5_table_cm_W2V_BERT = pd.crosstab(dtf_top5_pivot[("predicted", "W2V")], dtf_top5_pivot[("predicted", "BERT")]).reset_index()
dtf_top5_table_cm_W2V_BERT.columns = ["WORD2VEC/DISTILBERT", "0orthodox", "1heterodox"]

dtf_top5_pivot["num_heterodox_predictions"] = dtf_top5_pivot[("predicted_bin", "TFIDF")] + dtf_top5_pivot[("predicted_bin", "W2V")] + dtf_top5_pivot[("predicted_bin", "BERT")]
dtf_top5_pivot["avg_het_score"] = (dtf_top5_pivot[("predicted_prob", "TFIDF")] + dtf_top5_pivot[("predicted_prob", "W2V")] + dtf_top5_pivot[("predicted_prob", "BERT")]) / 3

dtf_top5_table_num_of_het_pred_temp = dtf_top5_pivot["num_heterodox_predictions"].value_counts()

dtf_top5_table_num_of_het_pred = pd.DataFrame({"Number of heterodox classifications": dtf_top5_table_num_of_het_pred_temp.index,
                                               "Count of articles": dtf_top5_table_num_of_het_pred_temp})



if save_tables:
    with open(data_path + "/" + plots_tables_folder + "/top5_table_cm_TFIDF_W2V.tex", 'w') as tf:
        tf.write(dtf_top5_table_cm_TFIDF_W2V.to_latex(index=False,
                                                       escape = False,
                                                       column_format="lcc"))

    with open(data_path + "/" + plots_tables_folder + "/top5_table_cm_TFIDF_BERT.tex", 'w') as tf:
        tf.write(dtf_top5_table_cm_TFIDF_BERT.to_latex(index=False,
                                                       escape = False,
                                                       column_format="lcc"))

    with open(data_path + "/" + plots_tables_folder + "/top5_table_cm_W2V_BERT.tex", 'w') as tf:
        tf.write(dtf_top5_table_cm_W2V_BERT.to_latex(index=False,
                                                       escape = False,
                                                       column_format="lcc"))

    with open(data_path + "/" + plots_tables_folder + "/top5_table_num_of_het_pred.tex", 'w') as tf:
        tf.write(dtf_top5_table_num_of_het_pred.to_latex(index=False,
                                                       escape = False,
                                                       column_format="cc"))

len(dtf_top5_pivot[dtf_top5_pivot["avg_het_score"] > 0.5])

if save_plots:

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=dtf_top5_pivot[("predicted_prob","TFIDF")])
    plt.xlabel("Heterodoxy score", fontsize=15)
    plt.ylabel("Article count", fontsize=15)
    plt.savefig(data_path + "/" + plots_tables_folder + "/top5_plot_histogram_TFIDF_heterodoxy_score")
    plt.close()

    '''
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=dtf_top5_pivot["avg_het_score"])
    ax.set_title("Distribution of average heterodoxy score of articles", fontsize=15)
    plt.savefig(data_path + "/" + plots_tables_folder + "/top5_plot_histogram_average_heterodoxy_score")
    plt.close()

    fig, ax = plt.subplots(figsize= (10,6))
    sns.histplot(data = dtf_top5_pivot[('predicted_prob', 'TFIDF')], color = "r", label = "TFIDF", element = "poly", alpha = 0).set(yscale = "log", xlim = ((0.05,0.95)))
    sns.histplot(data = dtf_top5_pivot[('predicted_prob', 'W2V')], color = "g", label = "W2V", element = "poly", alpha = 0).set(yscale = "log", xlim = ((0.05,0.95)))
    sns.histplot(data = dtf_top5_pivot[('predicted_prob', 'BERT')], color = "b", label = "BERT", element = "poly", alpha = 0).set(yscale = "log", xlim = ((0.05,0.95)))
    sns.histplot(data = dtf_top5_pivot["avg_het_score"], color = "y", label = "Average", element = "poly", alpha = 0).set(yscale = "log", xlim = ((0.05,0.95)))
    ax.set_title("Distribution of heterodoxy score of articles by model", fontsize = 15)
    plt.legend()
    plt.savefig(data_path + "/" + plots_tables_folder + "/top5_plot_histogram_average_heterodoxy_score")
    plt.close()

    fig, ax = plt.subplots(figsize= (10,6))
    sns.distplot(dtf_top5_pivot[('predicted_prob', 'TFIDF')], hist = False, kde = True, color = "r", label = "TFIDF").set(xlim = ((0.4,1)), ylim = ((0,0.5)) )
    sns.distplot(dtf_top5_pivot[('predicted_prob', 'W2V')], hist = False, kde = True, color = "b", label = "W2V").set(xlim = ((0.4,1)), ylim = ((0,0.5)) )
    sns.distplot(dtf_top5_pivot[('predicted_prob', 'BERT')], hist = False, kde = True, color = "g", label = "BERT").set(xlim = ((0.4,1)), ylim = ((0,0.5)) )
    ax.set_title("Distribution of heterodoxy score of articles by model", fontsize = 15)
    plt.legend()
    plt.savefig(data_path + "/" + plots_tables_folder + "/top5_plot_histogram_average_heterodoxy_score")
    plt.close()
    '''

# most heterodox
##############

dtf_top5_pivot = dtf_top5_pivot.applymap(lambda s: s.lower() if type(s) == str else s)

dtf_top5_tfidf_heterodox20 = dtf_top5_pivot[[("predicted_prob", "TFIDF"),
                                             ("journal", ""),
                                             ("pubyear", ""),
                                             ("author", ""),
                                             ("title", "")]].sort_values(by=("predicted_prob", "TFIDF"), ascending=False)[0:20].round(2)

colnames = ["Het. score", "Journal", "Year", "Authors", "Title"]
dtf_top5_tfidf_heterodox20.columns = colnames
dtf_top5_tfidf_heterodox20["Year"] = dtf_top5_tfidf_heterodox20["Year"].astype(int)
dtf_top5_tfidf_heterodox20["Authors"].str.replace("&","")

dtf_top5_tfidf_orthodox20 = dtf_top5_pivot[[("predicted_prob", "TFIDF"),
                                            ("journal", ""),
                                            ("pubyear", ""),
                                            ("author", ""),
                                            ("title", "")]].sort_values(by=("predicted_prob", "TFIDF"), ascending=True)[0:20].round(5)

dtf_top5_tfidf_orthodox20.columns = colnames
dtf_top5_tfidf_orthodox20["Year"] = dtf_top5_tfidf_orthodox20["Year"].astype(int)


if save_tables:
    with pd.option_context("max_colwidth", 1000):
        with open(data_path + "/" + plots_tables_folder + "/top5_table_top_20_most_heterodox.tex", 'w') as tf:
            tf.write(dtf_top5_tfidf_heterodox20.to_latex(index=False,
                                                       escape = False,
                                                       #column_format="clcll"))
                                                       column_format='p{0.7cm}p{3.5cm}p{1cm}p{2.5cm}p{5cm}'))

        with open(data_path + "/" + plots_tables_folder + "/top5_table_top_20_most_orthodox.tex", 'w') as tf:
            tf.write(dtf_top5_tfidf_orthodox20.to_latex(index=False,
                                                       escape = False,
                                                       #column_format="clcll"))
                                                        column_format='p{0.7cm}p{3.5cm}p{1cm}p{2.5cm}p{5cm}'))

# time series
########

bins = np.arange(1990, 2025, 5).tolist()
dtf_top5['pubyear_bin'] = pd.cut(dtf_top5['pubyear'], bins).astype("string")

dtf_top5_tfidf_agg = dtf_top5[dtf_top5["model"] == "TFIDF"].groupby(["journal", "pubyear_bin"]).mean(["predicted_bin", "predicted_prob", "rank"])
dtf_top5_tfidf_agg.reset_index(level=["journal", "pubyear_bin"], inplace=True)
dtf_top5_tfidf_agg = dtf_top5_tfidf_agg[["journal","pubyear_bin","predicted_bin","predicted_prob"]]
dtf_top5_tfidf_agg.columns = ["Journal","Years of publication","Prop. of heterodox articles","Avg. heterodoxy score",]

if save_plots:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=dtf_top5_tfidf_agg[["Years of publication", "Journal", "Prop. of heterodox articles"]],
                 x="Years of publication",
                 y="Prop. of heterodox articles",
                 hue="Journal")
    plt.xlabel("Years of publication", fontsize=15)
    plt.ylabel("Prop. of heterodox articles", fontsize=15)
    plt.savefig(data_path + "/" + plots_tables_folder + "/top5_plot_heterodox_articles_timeline")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=dtf_top5_tfidf_agg[["Years of publication", "Journal", "Avg. heterodoxy score"]],
                 x="Years of publication",
                 y="Avg. heterodoxy score",
                 hue="Journal")
    plt.xlabel("Years of publication", fontsize=15)
    plt.ylabel("Avg. heterodoxy score", fontsize=15)
    plt.savefig(data_path + "/" + plots_tables_folder + "/top5_plot_heterodoxy_score_timeline")
    plt.close()














#############################
#training data
#############

training_data = pd.read_csv(data_path + "/WOS_lee_heterodox_und_samequality_new_preprocessed_2.csv")
training_data = training_data.applymap(lambda s: s.lower() if type(s) == str else s)
training_data = training_data.applymap(lambda s: s.replace("0samequality", "0orthodox") if type(s) == str else s)
training_data = training_data.rename(columns={"y":"label"}).copy()

bins = np.arange(1990, 2025, 5).tolist()
training_data['pubyear_bin'] = pd.cut(training_data['Publication Year'], bins).astype("string")

training_data_agg = training_data[["pubyear_bin", "label", "Journal"]].groupby(["pubyear_bin", "label"]).size().reset_index()
training_data_agg.columns = ["pubyear_bin", "label", "num_articles"]

training_data_agg_pivot = training_data_agg.pivot(index="pubyear_bin",
                                                  columns="label",
                                                  values="num_articles").reset_index()
training_data_agg_pivot.columns = ["pubyear_bin", "orthodox_articles", "heterodox_articles"]
training_data_agg_pivot["proportion_heterodox"] = training_data_agg_pivot["heterodox_articles"] / (training_data_agg_pivot["heterodox_articles"] + training_data_agg_pivot["orthodox_articles"])

if save_plots:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=training_data_agg,
                 x="pubyear_bin",
                 y="num_articles",
                 hue="label")
    plt.xlabel("Years of publication", fontsize = 15)
    plt.ylabel("Number of articles", fontsize = 15)
    plt.savefig(data_path + "/" + plots_tables_folder + "/training_plot_timeline_num_articles")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=training_data_agg_pivot,
                 x="pubyear_bin",
                 y="proportion_heterodox")
    plt.xlabel("Years of publication", fontsize = 15)
    plt.ylabel("Proportion of heterodox articles", fontsize = 15)
    plt.savefig(data_path + "/" + plots_tables_folder + "/training_plot_timeline_proportion_heterodox")
    plt.close()


training_data_agg_journals = training_data.groupby(["Journal", "label"]).agg({"text":"count"}).reset_index()
training_data_agg_journals.columns = ["journal","label","num_articles"]
training_data_agg_journals.sort_values(["label","journal"], inplace = True)

journals_lee = pd.read_csv(data_path + "/table_lee_2010.csv")
journals_lee = journals_lee.applymap(lambda s: s.lower() if type(s) == str else s)
journals_lee.columns = [i.lower() for i in journals_lee.columns]

training_data_agg_journals = pd.merge(training_data_agg_journals, journals_lee, how = "left", on = ["journal"])

training_data_agg_journals["weighted_jqes"] = training_data_agg_journals["num_articles"] * training_data_agg_journals["jqes"]

training_data_agg_labels = training_data_agg_journals[["journal",
                                                      "label_x",
                                                      "num_articles",
                                                      "jqes",
                                                      "weighted_jqes"]].groupby("label_x").agg({"num_articles":"sum",
                                                                                               "jqes":"mean",
                                                                                               "weighted_jqes":"sum"}).reset_index()
training_data_agg_labels["weighted_jqes"] = training_data_agg_labels["weighted_jqes"] / training_data_agg_labels["num_articles"]

training_data_agg_labels.columns = ["label", "num_articles", "avg_jqes_journals", "avg_jqes_articles"]


training_data_agg_journals_latex = training_data_agg_journals[["journal",
                                                              "label_x",
                                                              "num_articles",
                                                              "jqes"]].round(2)

training_data_agg_journals_latex["journal"] = training_data_agg_journals_latex["journal"]

training_data_agg_journals_latex.columns = ["journal", "label", "num_articles", "jqes_rating"]

training_data_agg_journals_latex_orthodox = training_data_agg_journals_latex[training_data_agg_journals_latex["label"] == "0orthodox"][["journal","num_articles", "jqes_rating"]].sort_values("jqes_rating", ascending = False)
training_data_agg_journals_latex_heterodox = training_data_agg_journals_latex[training_data_agg_journals_latex["label"] == "1heterodox"][["journal","num_articles", "jqes_rating"]].sort_values("jqes_rating", ascending = False)



if save_tables:

    with open(data_path + "/" + plots_tables_folder + "/training_journal_list_orthodox.tex", 'w') as tf:
        tf.write(training_data_agg_journals_latex_orthodox.to_latex(index=False,
                                                                   escape = False,
                                                                   column_format="lcc"))

    with open(data_path + "/" + plots_tables_folder + "/training_journal_list_heterodox.tex", 'w') as tf:
        tf.write(training_data_agg_journals_latex_heterodox.to_latex(index=False,
                                                                   escape = False,
                                                                   column_format="lcc"))


    with open(data_path + "/" + plots_tables_folder + "/training_label_overview.tex", 'w') as tf:
        tf.write(training_data_agg_labels.round(2)
                                        .to_latex(index=False,
                                                  escape=False,
                                                  column_format="lccc"))


if save_plots:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.distplot(training_data_agg_journals_latex[training_data_agg_journals_latex["label"]=="0orthodox"]["jqes_rating"], color = "r", hist = False, kde = True, label = "0orthodox")
    sns.distplot(training_data_agg_journals_latex[training_data_agg_journals_latex["label"]=="1heterodox"]["jqes_rating"], color = "b", hist = False, kde = True, label = "1heterodox")
    plt.xlabel("Heterodoxy adjusted rating (JQES)", fontsize = 15)
    plt.ylabel("Density", fontsize=15)
    plt.legend()
    plt.savefig(data_path + "/" + plots_tables_folder + "/training_distribution_journal_ratings")
    plt.close()














#######################
# Top5 raw
#########

top5_data = pd.read_csv(data_path + "/WOS_top5_new_preprocessed_2.csv")

bins = np.arange(1990, 2025, 5).tolist()
top5_data['pubyear_bin'] = pd.cut(top5_data['Publication Year'], bins).astype("string")

top5_data_agg = top5_data[["Journal", "pubyear_bin"]].groupby(["Journal", "pubyear_bin"]).size().reset_index()

top5_data_agg.columns = ["journal","pubyear_bin","num_articles"]

if save_plots:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=top5_data_agg,
                 x="pubyear_bin",
                 y="num_articles",
                 hue="journal")
    plt.xlabel("Years of publication", fontsize = 15)
    plt.ylabel("Number of articles", fontsize=15)
    plt.savefig(data_path + "/" + plots_tables_folder + "/top5_timeline_articles")
    plt.close()


'''
top5_data_agg_latex = top5_data_agg.groupby("journal").sum("num_articles")


if save_tables:
    with open(data_path + "/" + plots_tables_folder + "/top5_table_articles_by_journal.tex", 'w') as tf:
        tf.write(top5_data_agg_latex.to_latex(caption="Number of articles by journal",
                                                         label="top5_table_articles_by_journal",
                                                         index=True))
'''

# authors
#####################

authors = [i.replace(" ", "") for s in [i.split(";") for i in dtf_top5["author"]] for i in s]

authors_unique = list(set(authors))

authors_list = [i.split(",") for i in authors_unique]

sum([len(i) == 1 for i in authors_list])
sum([len(i) == 2 for i in authors_list])
sum([len(i) == 3 for i in authors_list])

authors_list_clean = [i for i in authors_list if len(i) == 2]
authors_clean = [i[0] + "," + i[1] for i in authors_list_clean]
authors_clean = [i.lower() for i in authors_clean]

authors_final = [i.lower() for i in [i[0] + "," + i[1][0] for i in authors_list_clean]]

authors = [i.replace(" ", "").lower() for i in dtf_top5_pivot[('author', '')]]

map = dict(zip(authors_clean, authors_final))

for old, new in zip(authors_clean, authors_final):
    # print(old + "-->" + new)
    authors = [i.replace(old, new) for i in authors]

dtf_top5_pivot["author"] = authors
dtf_top5_pivot["author_list"] = [i.split(";") for i in authors]

dtf_top5_pivot_long = dtf_top5_pivot.explode("author_list", ignore_index=True)
dtf_top5_pivot_long["ind_author"] = [i for s in [i.split(";") for i in authors] for i in s]
dtf_top5_pivot_long.sort_values("ind_author", inplace=True)

numcols = [("predicted_prob", "TFIDF"),
           ("predicted_prob", "W2V"),
           ("predicted_prob", "BERT"),
           ("predicted_bin", "TFIDF"),
           ("predicted_bin", "W2V"),
           ("predicted_bin", "BERT"),
           ("rank", "TFIDF"),
           ("rank", "W2V"),
           ("rank", "BERT"),
           ("times_cited", "")]

for i in numcols:
    dtf_top5_pivot_long[i] = pd.to_numeric(dtf_top5_pivot_long[i], errors='coerce')

dtf_top5_pivot_long[("author_list", "")] = dtf_top5_pivot_long["author_list"].astype(object)

d = {"predicted_prob": "mean_predicted_prob", "predicted_bin": "sum_predicted_bin"}
dtf_top5_author_ranks = dtf_top5_pivot_long[["ind_author", ("predicted_prob", "TFIDF"), ("times_cited", ""), ("WOS_ID", "")]]
dtf_top5_author_ranks = dtf_top5_author_ranks.groupby("ind_author").agg({("predicted_prob", "TFIDF"): 'mean',
                                                                         ("times_cited", ""): 'sum',
                                                                         ("WOS_ID", ""): 'count'}).sort_values(("predicted_prob", "TFIDF"), ascending=False)

dtf_top5_author_citations = dtf_top5_pivot_long[["ind_author", ("predicted", "TFIDF"), ("times_cited", ""), ("WOS_ID", "")]]
dtf_top5_author_citations = dtf_top5_author_citations.groupby(["ind_author", ("predicted", "TFIDF")]).agg({("times_cited", ""): 'sum',
                                                                                                           ("WOS_ID", ""): 'count'})
dtf_top5_author_citations.reset_index(level=["ind_author", ("predicted", "TFIDF")], inplace=True)
dtf_top5_author_citations = dtf_top5_author_citations.pivot(index="ind_author",
                                                            columns=[("predicted", "TFIDF")],
                                                            values=[("times_cited", ""), ("WOS_ID", "")])

dtf_top5_authors = dtf_top5_author_ranks.join(dtf_top5_author_citations)

dtf_top5_authors.columns = ["predicted_prob_tfidf", "total_times_cited", "total_publications", "orthodox_citations", "heterodox_citations", "orthodox_publications", "heterodox_publications"]

# citations
################

dtf_top5_authors["orthodox_rel_citations"] = dtf_top5_authors["orthodox_citations"] / dtf_top5_authors["orthodox_publications"]
dtf_top5_authors["heterodox_rel_citations"] = dtf_top5_authors["heterodox_citations"] / dtf_top5_authors["heterodox_publications"]
dtf_top5_authors["heterodox_citation_index"] = dtf_top5_authors["heterodox_rel_citations"] / dtf_top5_authors["orthodox_rel_citations"]
dtf_top5_authors["weighted_prop_of_het_citations"] = dtf_top5_authors["heterodox_rel_citations"] / (dtf_top5_authors["heterodox_rel_citations"] + dtf_top5_authors["orthodox_rel_citations"])

dtf_top5_authors_atleast5 = dtf_top5_authors[dtf_top5_authors["total_publications"] > 4]

dtf_top5_authors_atleast5_onehet = dtf_top5_authors_atleast5[dtf_top5_authors_atleast5["heterodox_citations"] > 0]

dtf_top5_authors_atleast5_onehet["heterodox_citation_index"].median()
dtf_top5_authors_atleast5_onehet["weighted_prop_of_het_citations"].mean()
st.ttest_1samp(dtf_top5_authors_atleast5_onehet["weighted_prop_of_het_citations"], popmean=0.5)

st.t.interval(alpha=0.99,
              df=len(dtf_top5_authors_atleast5_onehet["weighted_prop_of_het_citations"]) - 1,
              loc=np.mean(dtf_top5_authors_atleast5_onehet["weighted_prop_of_het_citations"]),
              scale=st.sem(dtf_top5_authors_atleast5_onehet["weighted_prop_of_het_citations"]))

if save_plots:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=dtf_top5_authors_atleast5_onehet["weighted_prop_of_het_citations"],
                 kde=True)
    plt.xlabel("Weighted proportion of heterodox citations", fontsize=15)
    plt.ylabel("Author count", fontsize=15)
    plt.axvline(np.mean(dtf_top5_authors_atleast5_onehet["weighted_prop_of_het_citations"]), color="r")
    plt.savefig(data_path + "/" + plots_tables_folder + "/top5_plot_weighted_prop_of_het_citations")
    plt.close()

# heterodox publications
################

dtf_top5_authors_atleast5_onehet["prop_of_heterodox_publications"] = dtf_top5_authors_atleast5_onehet["heterodox_publications"] / dtf_top5_authors_atleast5_onehet["total_publications"]
#dtf_top5_authors_mosthet_count = dtf_top5_authors_atleast5_onehet[["total_publications", "heterodox_publications", "prop_of_heterodox_publications"]].sort_values(["heterodox_publications", "prop_of_heterodox_publications"], ascending=False)
dtf_top5_authors_mosthet_prop = dtf_top5_authors_atleast5_onehet[["total_publications", "heterodox_publications", "prop_of_heterodox_publications"]].sort_values(["prop_of_heterodox_publications", "heterodox_publications"], ascending=False)
dtf_top5_authors_mosthet_prop.reset_index(inplace = True)

#dtf_top5_authors_mosthet_count = dtf_top5_authors_mosthet_count[dtf_top5_authors_mosthet_count["heterodox_publications"] > 4]
dtf_top5_authors_mosthet_prop = dtf_top5_authors_mosthet_prop[dtf_top5_authors_mosthet_prop["prop_of_heterodox_publications"] > 0.4].round(2)
dtf_top5_authors_mosthet_prop["heterodox_publications"] = dtf_top5_authors_mosthet_prop["heterodox_publications"].astype(int)
dtf_top5_authors_mosthet_prop.columns = ["Author","Total publications","Heterodox publications","Proportion of heterodox publications"]

if save_tables:
    with open(data_path + "/" + plots_tables_folder + "/top5_table_most_heterodox_proportion.tex", 'w') as tf:
        tf.write(dtf_top5_authors_mosthet_prop.to_latex(index=False,
                                                  escape=False,
                                                  column_format="lccc"))

print("huh")
'''
if save_tables:
 
    with open(data_path + "/" + plots_tables_folder + "/top5_table_most_heterodox_count.tex", 'w') as tf:
        tf.write(dtf_top5_authors_mosthet_count.to_latex(caption="All authors with at lest 5 heterodox articles published in the Top 5 Journals (1990-2020)",
                                                         label="top5_table_most_heterodox_count",
                                                         index=True))


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















##########
# WEIGHTS
##########

dtf_weights_tfidf = load_data(information="7Weights_All_Data", model="TFIDF", results_folder=results_folder)

dtf_weights_tfidf_pivot = dtf_weights_tfidf.pivot(index=["words", "weights"],
                                                  columns=["prediction"],
                                                  values="counts")

dtf_weights_tfidf_pivot.reset_index(["words", "weights"], inplace=True)
dtf_weights_tfidf_pivot.fillna(0, inplace=True)
dtf_weights_tfidf_pivot.columns = ["word", "weight", "orthodox_count", "heterodox_count"]
dtf_weights_tfidf_pivot = dtf_weights_tfidf_pivot[dtf_weights_tfidf_pivot["weight"] != 0].copy()
dtf_weights_tfidf_pivot["total_count"] = dtf_weights_tfidf_pivot["orthodox_count"] + dtf_weights_tfidf_pivot["heterodox_count"]
dtf_weights_tfidf_pivot["weighted_orthodox_count"] = (dtf_weights_tfidf_pivot["weight"] * dtf_weights_tfidf_pivot["orthodox_count"]) / dtf_weights_tfidf_pivot["total_count"]
dtf_weights_tfidf_pivot["weighted_heterodox_count"] = (dtf_weights_tfidf_pivot["weight"] * dtf_weights_tfidf_pivot["heterodox_count"]) / dtf_weights_tfidf_pivot["total_count"]
dtf_weights_tfidf_pivot["weighted_total_count"] = dtf_weights_tfidf_pivot["weight"] * dtf_weights_tfidf_pivot["total_count"]
dtf_weights_tfidf_pivot.sort_values("weight", inplace=True)

if save_plots:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=dtf_weights_tfidf_pivot["weight"], kde=True)
    plt.xlabel("Logistic regression weight", fontsize=15)
    plt.ylabel("Word count", fontsize=15)
    plt.axvline(np.mean(dtf_weights_tfidf_pivot["weight"]), color="r")
    plt.savefig(data_path + "/" + plots_tables_folder + "/weights_histogram_tfidf")
    plt.close()

dtf_weights_tfidf_top20_orthodox = dtf_weights_tfidf_pivot[["word", "weight", "total_count"]][0:20].copy()
dtf_weights_tfidf_top20_orthodox = dtf_weights_tfidf_top20_orthodox.round(2)
dtf_weights_tfidf_top20_orthodox.columns = ["Word", "Weight", "Count"]
dtf_weights_tfidf_top20_orthodox["Count"] = dtf_weights_tfidf_top20_orthodox["Count"].astype(int)

dtf_weights_tfidf_top20_heterodox = dtf_weights_tfidf_pivot[["word", "weight", "total_count"]][len(dtf_weights_tfidf_pivot) - 20:len(dtf_weights_tfidf_pivot)].sort_values("weight", ascending=False).copy()
dtf_weights_tfidf_top20_heterodox = dtf_weights_tfidf_top20_heterodox.round(2)
dtf_weights_tfidf_top20_heterodox.columns = ["Word", "Weight", "Count"]
dtf_weights_tfidf_top20_heterodox["Count"] = dtf_weights_tfidf_top20_heterodox["Count"].astype(int)


if save_tables:
    with open(data_path + "/" + plots_tables_folder + "/weights_top20_orthodox_words.tex", 'w') as tf:
        tf.write(dtf_weights_tfidf_top20_orthodox.to_latex(index=False,
                                                           escape = False,
                                                           column_format="lcc"))

    with open(data_path + "/" + plots_tables_folder + "/weights_top20_heterodox_words.tex", 'w') as tf:
        tf.write(dtf_weights_tfidf_top20_heterodox.to_latex(index=False,
                                                           escape = False,
                                                           column_format="lcc"))

dtf_weights_tfidf_pivot['percentile'] = pd.qcut(dtf_weights_tfidf_pivot['weight'], 100, labels=False)

selected_words = ["imperialism",
                  "power",
                  "utility",
                  "marginal",
                  "inelastic",
                  "monopolist",
                  "supply",
                  "equilibrium",
                  "rational",
                  "maximize",
                  "opportunity",
                  "cost",
                  "tradeoff",
                  "substitute",
                  "rent",
                  "optimum",
                  "optimal",
                  "preference",
                  "elasticity",
                  "market",
                  "loanable",
                  "demand",
                  "institution",
                  "innovation",
                  "destruction",
                  "structural",
                  "demand",
                  "unemployment",
                  "gender",
                  "woman",
                  "class",
                  "power",
                  "exploitation",
                  "evolution",
                  "dynamic",
                  "system",
                  "agent",
                  "behaviour",
                  "psychology",
                  "mental",
                  "unemployment",
                  "inflation",
                  "mental",
                  "optimization",
                  "scarcity",
                  "growth",
                  "profit",
                  "technology",
                  "technological",
                  "finance"]



neglected = ["imperialism",
                  "power",
                  "anti",
             "censorship",
             "colonization"]

jargon = ["loanable",
          "monopolist",
          "inelastic",
          "marginal",
          "utility",
          "demand",
          "price",
          "supply",
          "market"]

neoclassical = ["equilibrium",
                  "rational",
                  "maximize",
                  "opportunity",
                  "tradeoff",
                  "substitute",
                  "rent",
                  "optimum",
                  "optimal",
                  "preference",
                  "elasticity",
                  "optimization",
                  "scarcity",
                  "growth",
                  "profit"]

heterodox = ["institution",
              "innovation",
              "destruction",
              "structural",
              "demand",
              "unemployment",
              "gender",
              "woman",
              "class",
              "power",
              "exploitation",
              "evolution",
              "dynamic",
              "system",
              "agent",
              "behaviour",
              "psychology",
              "mental",
              "unemployment",
              "mental"]



dtf_weights_tfidf_pivot_short = dtf_weights_tfidf_pivot[["word", "weight", "total_count", "percentile"]].round(2)
dtf_weights_tfidf_pivot_short.columns = ["Word", "Weight", "Count", "Percentile of weight"]
dtf_weights_tfidf_pivot_short["Count"] = dtf_weights_tfidf_pivot_short["Count"].astype(int)

dtf_weights_tfidf_pivot_jargon = dtf_weights_tfidf_pivot_short[dtf_weights_tfidf_pivot_short.Word.isin(jargon)].copy()
dtf_weights_tfidf_pivot_neglected = dtf_weights_tfidf_pivot_short[dtf_weights_tfidf_pivot_short.Word.isin(neglected)].copy()
dtf_weights_tfidf_pivot_neoclassical = dtf_weights_tfidf_pivot_short[dtf_weights_tfidf_pivot_short.Word.isin(neoclassical)].copy()
dtf_weights_tfidf_pivot_heterodox = dtf_weights_tfidf_pivot_short[dtf_weights_tfidf_pivot_short.Word.isin(heterodox)].copy()



if save_tables:
    with open(data_path + "/" + plots_tables_folder + "/weights_tfidf_jargon.tex", 'w') as tf:
        tf.write(dtf_weights_tfidf_pivot_jargon.to_latex(index=False,
                                                           escape = False,
                                                           column_format="lccc"))

    with open(data_path + "/" + plots_tables_folder + "/weights_tfidf_neglected.tex", 'w') as tf:
        tf.write(dtf_weights_tfidf_pivot_neglected.to_latex(index=False,
                                                           escape = False,
                                                           column_format="lccc"))

    with open(data_path + "/" + plots_tables_folder + "/weights_tfidf_neoclassical.tex", 'w') as tf:
        tf.write(dtf_weights_tfidf_pivot_neoclassical.to_latex(index=False,
                                                           escape = False,
                                                           column_format="lccc"))

    with open(data_path + "/" + plots_tables_folder + "/weights_tfidf_heterodox.tex", 'w') as tf:
        tf.write(dtf_weights_tfidf_pivot_heterodox.to_latex(index=False,
                                                           escape = False,
                                                           column_format="lccc"))






if save_tables:
    with open(data_path + "/" + plots_tables_folder + "/weights_tfidf_selected_words.tex", 'w') as tf:
        tf.write(dtf_weights_tfidf_pivot_selection.to_latex(caption="weigths and percentile rank of selected words",
                                                            label="weights_tfidf_selected_words",
                                                            index=False))









# weights top 5
###############

dtf_weights_top5_tfidf = load_data(information="6Weights_Top5", model="TFIDF", results_folder=results_folder)

dtf_weights_top5_tfidf_pivot = dtf_weights_top5_tfidf.pivot(index=["words", "weights"],
                                                            columns=["prediction"],
                                                            values="counts")

dtf_weights_top5_tfidf_pivot.reset_index(["words", "weights"], inplace=True)
dtf_weights_top5_tfidf_pivot.fillna(0, inplace=True)
dtf_weights_top5_tfidf_pivot.columns = ["word", "weight", "orthodox_count", "heterodox_count"]
dtf_weights_top5_tfidf_pivot = dtf_weights_top5_tfidf_pivot[dtf_weights_top5_tfidf_pivot["weight"] != 0].copy()
dtf_weights_top5_tfidf_pivot["total_count"] = dtf_weights_top5_tfidf_pivot["orthodox_count"] + dtf_weights_top5_tfidf_pivot["heterodox_count"]
dtf_weights_top5_tfidf_pivot["weighted_orthodox_count"] = (dtf_weights_top5_tfidf_pivot["weight"] * dtf_weights_top5_tfidf_pivot["orthodox_count"]) / dtf_weights_top5_tfidf_pivot["total_count"]
dtf_weights_top5_tfidf_pivot["weighted_heterodox_count"] = (dtf_weights_top5_tfidf_pivot["weight"] * dtf_weights_top5_tfidf_pivot["heterodox_count"]) / dtf_weights_top5_tfidf_pivot["total_count"]
dtf_weights_top5_tfidf_pivot["weighted_total_count"] = dtf_weights_top5_tfidf_pivot["weight"] * dtf_weights_top5_tfidf_pivot["total_count"]
dtf_weights_top5_tfidf_pivot.sort_values("weighted_total_count", inplace=True)
len(dtf_weights_top5_tfidf_pivot[abs(dtf_weights_top5_tfidf_pivot["weighted_total_count"])>1000])
len(dtf_weights_top5_tfidf_pivot[dtf_weights_top5_tfidf_pivot["weighted_total_count"]>1000])
len(dtf_weights_top5_tfidf_pivot[dtf_weights_top5_tfidf_pivot["weighted_total_count"]<-1000])


if save_plots:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.distplot(dtf_weights_top5_tfidf_pivot["weighted_total_count"], hist=True, kde=False)
    ax.set_title("Distribution of the words counts weighted with the logistic regression weights in the TFIDF model", fontsize=15)
    plt.axvline(np.mean(dtf_weights_top5_tfidf_pivot["weighted_total_count"]), color="r")
    plt.savefig(data_path + "/" + plots_tables_folder + "/weighted_total_count_histogram_top5_tfidf")
    plt.close()

dtf_weights_top5_tfidf_top20_orthodox = dtf_weights_top5_tfidf_pivot[["word", "weighted_total_count", "weight" , "total_count"]][0:20].copy()
dtf_weights_top5_tfidf_top20_heterodox = dtf_weights_top5_tfidf_pivot[["word", "weighted_total_count", "weight" , "total_count"]][len(dtf_weights_top5_tfidf_pivot) - 20:len(dtf_weights_top5_tfidf_pivot)].sort_values("weighted_total_count", ascending=False).copy()

if save_tables:
    with open(data_path + "/" + plots_tables_folder + "/weights_top5_tfidf_top20_orthodox_words.tex", 'w') as tf:
        tf.write(dtf_weights_top5_tfidf_top20_orthodox.to_latex(caption="20 words with the smallest Logistic Regression weights in the TFIDF model",
                                                                label="weights_top5_tfidf_top20_orthodox_words",
                                                                index=False))

    with open(data_path + "/" + plots_tables_folder + "/weights_top5_tfidf_top20_heterodox_words.tex", 'w') as tf:
        tf.write(dtf_weights_top5_tfidf_top20_heterodox.to_latex(caption="20 words with the largest Logistic Regression weights in the TFIDF model",
                                                                 label="weights_top5_tfidf_top20_heterodox_words",
                                                                 index=False))



selected_words = ["imperialism",
                  "power",
                  "utility",
                  "marginal",
                  "inelastic",
                  "monopolist",
                  "supply",
                  "equilibrium",
                  "rational",
                  "maximize",
                  "opportunity",
                  "cost",
                  "tradeoff",
                  "substitute",
                  "rent",
                  "optimum",
                  "optimal",
                  "preference",
                  "elasticity",
                  "market",
                  "loanable",
                  "demand",
                  "institution",
                  "innovation",
                  "destruction",
                  "structural",
                  "demand",
                  "unemployment",
                  "gender",
                  "woman",
                  "class",
                  "power",
                  "exploitation",
                  "evolution",
                  "dynamic",
                  "system",
                  "agent",
                  "behaviour",
                  "psychology",
                  "mental",
                  "unemployment",
                  "inflation",
                  "mental",
                  "optimization",
                  "scarcity",
                  "growth",
                  "profit",
                  "technology",
                  "finance"]

dtf_weights_top5_tfidf_pivot_selection = dtf_weights_top5_tfidf_pivot[dtf_weights_top5_tfidf_pivot.word.isin(selected_words)][["word", "weighted_total_count", "weight", "total_count"]].copy()

if save_tables:
    with open(data_path + "/" + plots_tables_folder + "/weighted_counts_top5_tfidf_selected_words.tex", 'w') as tf:
        tf.write(dtf_weights_top5_tfidf_pivot.to_latex(caption="Weigthted counts of selected words",
                                                            label="weighted_counts_top5_tfidf_selected_words",
                                                            index=False))
















# individual authors
####################

import pickle
from lime import lime_text

input_file_name = "WOS_top5_new_preprocessed_2"

model_file_name = "3All_Data_TFIDF"

dtf = pd.read_csv(data_path + "/" + input_file_name + ".csv")

model_file_path = (data_path + "/models/" + model_file_name + ".pkl")

with open(model_file_path, 'rb') as file:
    model_loaded = pickle.load(file)


## select observation

dtf_top5_top_5_het = dtf_top5[dtf_top5["model_file_name"] == "3All_Data_TFIDF"].sort_values("predicted_prob", ascending = False)[0:20].copy()

explainer = lime_text.LimeTextExplainer(class_names=["0orthodox", "1heterodox"])

for i in range(5):
    id = dtf_top5_top_5_het.iloc[i]["WOS_ID"]
    with pd.option_context("max_colwidth", 10000):
        txt_instance_temp = str(dtf["text_clean"][dtf["UT (Unique WOS ID)"] == id])
        txt_instance_raw_temp = str(dtf["text"][dtf["UT (Unique WOS ID)"] == id])
    txt_instance = txt_instance_temp[6:len(txt_instance_temp) - 32]

    print(txt_instance_raw_temp)

    ## show explanation
    explained = explainer.explain_instance(txt_instance, model_loaded.predict_proba, num_features=100)
    explained.save_to_file(data_path + "/" + plots_tables_folder + "/explainer_most_heterodox_" + str(i) + ".html")


dtf_top5_piketty = dtf_top5_pivot_long[dtf_top5_pivot_long["ind_author"] == "piketty,t"]



explainer = lime_text.LimeTextExplainer(class_names=["0orthodox", "1heterodox"])



for i in range(len(dtf_top5_piketty)):
    id = dtf_top5_piketty.iloc[i][("WOS_ID","")].upper()
    with pd.option_context("max_colwidth", 10000):
        txt_instance_temp = str(dtf["text_clean"][dtf["UT (Unique WOS ID)"] == id])
        txt_instance_raw_temp = str(dtf["text"][dtf["UT (Unique WOS ID)"] == id])
    txt_instance = txt_instance_temp[6:len(txt_instance_temp)-32]

    print(txt_instance_raw_temp)

    ## show explanation
    explained = explainer.explain_instance(txt_instance, model_loaded.predict_proba, num_features=100)
    explained.save_to_file(data_path + "/" + plots_tables_folder + "/explainer_piketty_" + str(i) + ".html")





dtf_top5_goyal = dtf_top5_pivot_long[dtf_top5_pivot_long["ind_author"] == "goyal,s"]



explainer = lime_text.LimeTextExplainer(class_names=["0orthodox", "1heterodox"])



for i in range(len(dtf_top5_goyal)):
    id = dtf_top5_goyal.iloc[i][("WOS_ID","")].upper()
    with pd.option_context("max_colwidth", 10000):
        txt_instance_temp = str(dtf["text_clean"][dtf["UT (Unique WOS ID)"] == id])
        txt_instance_raw_temp = str(dtf["text"][dtf["UT (Unique WOS ID)"] == id])
    txt_instance = txt_instance_temp[6:max(7,len(txt_instance_temp)-32)]

    print(txt_instance_raw_temp)

    ## show explanation
    explained = explainer.explain_instance(txt_instance, model_loaded.predict_proba, num_features=100)
    explained.save_to_file(data_path + "/" + plots_tables_folder + "/explainer_goyal_" + str(i) + ".html")



dtf_top5_greenwood = dtf_top5_pivot_long[dtf_top5_pivot_long["ind_author"] == "greenwood,j"]



explainer = lime_text.LimeTextExplainer(class_names=["0orthodox", "1heterodox"])



for i in range(len(dtf_top5_greenwood)):
    id = dtf_top5_greenwood.iloc[i][("WOS_ID","")].upper()
    with pd.option_context("max_colwidth", 10000):
        txt_instance_temp = str(dtf["text_clean"][dtf["UT (Unique WOS ID)"] == id])
        txt_instance_raw_temp = str(dtf["text"][dtf["UT (Unique WOS ID)"] == id])
    txt_instance = txt_instance_temp[6:max(7,len(txt_instance_temp)-32)]

    print(txt_instance_raw_temp)

    ## show explanation
    explained = explainer.explain_instance(txt_instance, model_loaded.predict_proba, num_features=100)
    explained.save_to_file(data_path + "/" + plots_tables_folder + "/explainer_greenwood_" + str(i) + ".html")




dtf_top5_maggiori = dtf_top5_pivot_long[dtf_top5_pivot_long["ind_author"] == "maggiori,m"]



explainer = lime_text.LimeTextExplainer(class_names=["0orthodox", "1heterodox"])



for i in range(len(dtf_top5_maggiori)):
    id = dtf_top5_maggiori.iloc[i][("WOS_ID","")].upper()
    with pd.option_context("max_colwidth", 10000):
        txt_instance_temp = str(dtf["text_clean"][dtf["UT (Unique WOS ID)"] == id])
        txt_instance_raw_temp = str(dtf["text"][dtf["UT (Unique WOS ID)"] == id])
    txt_instance = txt_instance_temp[6:max(7,len(txt_instance_temp)-32)]

    print(txt_instance_raw_temp)

    ## show explanation
    explained = explainer.explain_instance(txt_instance, model_loaded.predict_proba, num_features=100)
    explained.save_to_file(data_path + "/" + plots_tables_folder + "/explainer_maggiori_" + str(i) + ".html")



