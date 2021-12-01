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
    plt.savefig(data_path + "/" + plots_tables_folder + "/model_selection_tfidf_auc_pr")
    plt.close()


    fig, ax = plt.subplots(figsize= (10,6))
    sns.lineplot(data = dtf_model_selection_tfidf[["number_relevant_features","min_df", "AUC_PR", "p_value_limit", "ngram_range"]],
                 x = "p_value_limit",
                 y = "number_relevant_features",
                 hue = "min_df",
                 style = "ngram_range").set(yscale = "log")
    ax.set_title("Number of features for term-frequency based classification systems", fontsize = 15)
    plt.savefig(data_path + "/" + plots_tables_folder + "/model_selection_tfidf_num_features")
    plt.close()
