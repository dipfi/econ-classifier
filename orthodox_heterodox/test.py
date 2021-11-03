import pandas as pd


dtf = pd.read_csv("C:/Users/Damian/kDrive/Main/Master/Masterarbeit/Ash/dev/data/WOS_lee_heterodox_und_samequality_preprocessed.csv")
dtf = dtf.sample(30000).copy()
dtf.to_csv("C:/Users/Damian/kDrive/Main/Master/Masterarbeit/Ash/dev/data/WOS_lee_heterodox_und_samequality_preprocessed_30000.csv")