# econ-classifier
This is the code base for the Master Thesis "The Dialects of Econospeak" by Damian Durrer (2021)

##Abstrac & Info:
"This thesis explores the differences in the language used in heterodox and orthodox economics journals with machine learning. We trained and evaluated various document classification models with a set of articles which were published in labeled journals between 1990 and 2020. The differences between the heterodox and orthodox dialect of economics are then explored by analyzing the feature weights in the models. We applied the models to the articles published in the top 5 economics journals during the same time, evaluate the proportion of heterodox articles by journal, and identify the most heterodox articles and authors. Various models from different categories of classification systems are fitted and compared: A benchmark model using a logistic regression classifier on a tf-idf weighted feature matrix, and two competitors using word2vec embeddings, BiLSTMs and DistilBert transformers. The benchmark model outperforms both competitors on the classification task. We achieve a reasonable performance in discriminating articles published in heterodox economic journals from articles published in orthodox economic journals using automated text classification systems. Our analysis indicates, that the proportion of heterodox articles in the top 5 journals has been declining over the past three decades."

The full text is available upon request.

All the data, tables and plots to use the code on can be accessed on:
https://drive.infomaniak.com/app/share/249519/d8dab04d-7ced-4f3a-a995-1916b3aa03a9
--> Download data and retain the folder structure to use the code without adjustments.

On the ETHZ Euler cluster, the data and code can be found under: /cluster/work/lawecon/Projects/Ash_Durrer/dev/"

Much of the code found here is based on the article by Di Pietro (2021):
https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794


##Prerequisits
  To run the code, the following set up is required:
    python 3.7
    cuda 11.1
    cudnn 8.1

  If run on the Euler Cluster at ETHZ, the required settings can be found in the notes folder under .bash_profile
     --> gcc/6.3.0 python_gpu/3.7.4 cuda/11.1.1 cudnn/8.1.0.77

  All other requirements can be loaded using the requirements.txt.
    --> pip install -r [path]/requirements.txt

  Create a file called "config.ini" and locate it in the "econ-classifier" main folder of the "econ-classifier"
    --> see "config_example.ini" as example

##structure
###orthodox_heterodox/
combine_excel_files.py:
  helper file to combine multiple excel files downloaded from Web of Science (uncommented)

preprocessing.py:
  - script to pre-process the files downloaded from Web of Science
  - comments are in the script

training_and_applying_models.py:
- main script used to select, train, evaluate and apply the models
- can be used along with the "REPRODUCE_*" files in the "Utils" folder to reproduce the results discussed in the thesis
- comments are in the script

analyze_results.py:
- script to produce the plots and tables in the Thesis

###Utils/
path_config: Configuring paths based on the config.ini file

utils_ortho_hetero: some functions regularly used in various scripts in the folder "orthodox_heterodox/"

REPRODUCE_*: parameter settings which can be used along with "orthodox_heterodox/training_and_applying_models.py": to reproduce the results from the thesis.
