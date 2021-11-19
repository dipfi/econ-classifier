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

import pandas as pd

def combine_all_excel_in_folder(folder_name = "WOS_clarivate_top5",
                          files_path = "/WOS_clarivate_top5/data",
                          file_ending = '.xls'
                          ):
    folder = str(data_path + files_path)
    files = os.listdir(folder)
    df = pd.DataFrame()
    for file in files:
        if file.endswith('.xls'):
            df = df.append(pd.read_excel(str(folder + "/" + file)), ignore_index=True)
    print(df.head())
    combined_file_path = str(folder + "/" + "combined_" + folder_name + "_new.csv")
    df.to_csv(combined_file_path, index = False)
    return combined_file_path

def combine_csv(input_names = ["combined_WOS_clarivate_lee_orthodox_samequality.csv",
                                  "combined_WOS_clarivate_lee_heterodox.csv"],
                   labels = ["0samequality",
                                "1heterodox"],
                   output_name = "WOS_lee_heterodox_und_samequality_new.csv",
                   ):

    # reading the files
    help_iterable = pd.DataFrame({"input": input_names,
                                 "labels": labels})
    df = pd.DataFrame()
    for index, row in help_iterable.iterrows():
        print(index)
        df_temp = pd.read_csv(str(data_path + "/" + row["input"]))
        df_temp["labels"] = row["labels"]
        df = df.append(df_temp, ignore_index=True)

    df.to_csv(str(data_path + "/" + output_name), index = False)

combined_file_path = combine_all_excel_in_folder()
combine_csv(input_names = [combined_file_path])