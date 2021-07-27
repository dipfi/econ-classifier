import time
tic = time.perf_counter()

import pandas as pd

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


if config['PATH']['data_path'][0] == "C":
    from tqdm import tqdm
    tqdm.pandas()


'''
INPUT PARAMETERS HERE
'''


############################################
file_name = "sample_for_damian"
data_orig = pd.read_csv(data_path + "/" + file_name + ".csv")
##########################################

from unidecode import unidecode

data_narrow = data_orig.loc[:,["journal","discipline"]]
journalcounts = data_narrow.groupby(["journal","discipline"]).size().reset_index(name = "count")

journalcounts["journal_unidecode"] = journalcounts["journal"].progress_apply(lambda x: unidecode(str(x).lower()))
heterodoxjournals = pd.read_csv(data_path + "/journals_short_list.csv")
heterodoxjournals["journal_unidecode"] = heterodoxjournals["journal_names"].progress_apply(lambda x: unidecode(str(x).lower()))
journalcounts_heterodox = journalcounts.loc[journalcounts["journal_unidecode"].isin(heterodoxjournals["journal_unidecode"]),:]
journalcounts_heterodox = journalcounts_heterodox.sort_values(["discipline", "count"])

journalcounts_heterodox.to_csv(data_path + '/sample_feat_heterodox_short_list.csv', index=False)

austrian = journalcounts.loc[journalcounts["journal_unidecode"].str.contains("cambridge")==True,:]