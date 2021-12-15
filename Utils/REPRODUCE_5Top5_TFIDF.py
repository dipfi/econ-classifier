'''
This script is part of the Master Thesis of Damian Durrer (SfS ETH Zürich), submission date 16 December 2021.

This script is part of the following project:
- Github: https://github.com/dipfi/econ-classifier
- Euler: /cluster/work/lawecon/Projects/Ash_Durrer/dev/scripts

The data for reproduction can be shared upon request:
- Alternatively for members of the LawEcon group at ETHZ
    -Euler: /cluster/work/lawecon/Projects/Ash_Durrer/dev/data

'''


import logging
import multiprocessing as mp
'''
----------------------------------------------------------
DEFAULT PARAMETERS
----------------------------------------------------------
'''
#######################################
logging_level = logging.INFO
cores = mp.cpu_count()
plot = 0
input_file_name = ""
text_field_clean = "text_clean"
text_field = "text"
label_field = "y"
journal_field = "Journal"
save_results = False
results_file_name = False
save_weights = False
current_model = False
min_df_list = []
p_value_limit_list = []
ngram_range_list = []
tfidf_classifier_list = []
embedding_vector_length_list = []
train_new = False
embedding_only = False
num_epochs_for_embedding_list = []
window_size_list = []
embedding_set = False
use_gigaword = False
use_embeddings = False
embedding_folder = "embeddings"
which_embeddings = False
num_epochs_for_classification_list = []
max_length_of_document_vector_w2v_list = []
classifier_loss_function_w2v_list = []
w2v_batch_size_list = []
small_model_list = []
bert_batch_size_list = []
bert_epochs_list = []
max_length_of_document_vector_bert_list = []
classifier_loss_function_bert_list = []
use_bert_feature_matrix = False
save_bert_feature_matrix = False
use_model = False
model_file_name = False
save_model = False
train_on_all = False
training_set = False
journal_split = False
use_reproducible_train_test_split = False
train_set_name = ""
test_set_name = ""
test_size = 0.1
num_journals = "all"
journal_list = []


#######################################


'''
settings
----------------------------------------------------------
'''
logging_level = logging.INFO
# Controlling the outputs printed in the console
#### Recommended: logging.INFO
#### Alternatives: logging.DEBUG ; logging.WARNING

cores = mp.cpu_count()
# Controlling the number of cores used in computations
#### Recommended: mp.cpu_count() // all cores
#### Alternatives: 1,2, ... // specify number of cores

plot = 0
# Controlling analytics plots to be created
#### Recommended: 0 // no plots
#### Alternatives: 1 // some plots : 2 // all plots


'''
input file
----------------------------------------------------------
'''
input_file_name = "WOS_top5_new_preprocessed_2"
## File needs to be located under the DATA_PATH and include the below specified columns (w/out ".csv"):
#### Recommended: Full Dataset: "WOS_lee_heterodox_und_samequality_new_preprocessed_2" // for model training and performance evaluation
#### Alternative:  Top 5 Dataset: "WOS_top5_new_preprocessed_2" // application
#### Alternative:  Short dataset to test code: "WOS_lee_heterodox_und_samequality_new_preprocessed_2_1000" // development

text_field_clean = "text_clean"  # "title" #"abstract"
## column name of the pre-processed abstracts in the input file // use SCRIPTS_PATH/preprocessing.py // for TFIDF and WORD2VEC
#### Recommended: 'text_clean'

text_field = "text"
## column name for the raw abstracts in the input file // for BERT
#### Recommended: 'text'

label_field = "y"
## column name for the labels {"0samequality", "1heterodox"} in the input file // for training, not required if pre-trained models are applied
#### Recommended: 'y'
#### Alternative: 'labels'

journal_field = "Journal"
## column name for the journals in the input file
#### Recommended: 'Journal'


'''
save results, models, weights and training-samlpes
----------------------------------------------------------
'''
save_results = True
## False if results should not be saved, True if they should be saved // OVERWRITES PREVIOUS FILES WITH THE SAME NAME

results_file_name = "REPRODUCTION_5Top5_TFIDF"
## Save the results to a file  in DATA_PATH/results // OVERWRITES PREVIOUS FILES WITH THE SAME NAME
#### Recommended: False // automatic name is set based on parameter choices
#### Alternative: e.g. "5TOP5_BERT" // Chose file name according to the settings ("TRAINING", "JOURNALS", "TOP5") and the models used ("TFIDF","W2V", "BERT")

save_weights = True
## save TFIDF weights to DATA_PATH/weights // name is automatically assigned accorting to the model name and input data name


'''
Set model hyperparameters and settings
----------------------------------------------------------
'''
current_model = "tfidf"
## choose which type of model to use: "tfidf", "w2v" or "bert"
#### Recommended: "tfidf" // tfidf weighting of features with subsequent classification (classifier can be selected below)
#### Alternative: "w2v" // word embeddings with subsequent LSTM classification; "bert" // bert transformer model

#### FOR ALL THE BELOW:
#### Provide parameters as list. If multiple parameters are provided: grid search over all combination is performed

if current_model == "tfidf":
    min_df_list = [5]
    ## choose the minimum document frequency for the terms (only terms with document frequency > min_df are included in the feature matrix
    #### Recommended: [5]
    #### Alternative: [3, 10]

    p_value_limit_list = [0.0]  # [0.8, 0.9, 0.95]
    ## choose the minimum p-value with which a term has to be correlated to the label (only terms with a p-value of larger than  p_value_limit are included)
    #### Recommended: [0.0] // no p-value limit applied
    #### Alternative: [0.9, 0.95] //

    ngram_range_list = [(1, 1)]
    ## choose range of n-grams to include as features
    #### Recommended: [1,1] // only words (1-grams)
    #### Alternative: [1,3] // 1-, 2-, and 3-grams are included; any other range is possible

    tfidf_classifier_list = ["LogisticRegression"]  # ["naive_bayes", "LogisticRegression", "RandomForestClassifier","GradientBoostingClassifier", "SVC"]
    ## choose a classifier
    #### Recommended: ["LogisticRegression"] // only words (1-grams)
    #### Alternative: ["naive_bayes", "RandomForestClassifier","GradientBoostingClassifier", "SVC"]

if current_model == "w2v":

    embedding_vector_length_list = [300]
    ## choose length of the embedding vectors (to train or load)
    #### Recommendation: [300] // 300-dimensional embeddings
    #### Alternative: [50, 150]

    train_new = True
    ##choose whether you want to train new embeddings

    embedding_only = False
    ## choose whether to only train the embeddings without the classification step (used to only store embeddings to use for classification later)
    #### Recommendation : False // perform classification after training the embeddings
    #### Alternative : True // only train embedding without classification (used to only store embeddings

    if train_new:
        num_epochs_for_embedding_list = [5]
        ## choose number of epochs to train the word embeddings
        #### Recommendation: [5]
        #### Alternative: [10,15]

        window_size_list = [12]
        ## choose window-size to use in for training the embeddings
        #### Recommendation: [12]
        #### Alternative: [4, 8]

        embedding_set = False
        ## choose whether or not to balance the training set before training the embeddings (independent of whether it is balanced for the classification)
        #### Recommendation: False // no balancing of the training set for the embeddings
        #### Alternative: "oversample" // apply random oversampling of the minority class;  "undersamling" // apply random undersampling of the majority class

    if not train_new:
        use_gigaword = False
        ## chose whether to use the pretrained embeddings from "glove-wiki-gigaword-[embedding_vector_length]d" --> IF TRUE, NO EMBEDDINGS WILL BE TRAINED ON YOUR DATA
        #### Recommended: False // dont use pretrained embeddings
        #### Alternative: True

        if not use_gigaword:
            use_embeddings = False
            ## choose your own pre-trained embeddings if True
            ####Recommended: False // train your own embeddings
            ####Alternative: True // use pretrained embeddigns --> specify source in "which_emneddings"

            if use_embeddings:
                embedding_folder = "embeddings"
                which_embeddings = False
                ##specify path where embeddings are stored (under DATA_PATH/[embedding_folder]/[which_embeddings]

    if not embedding_only:
        num_epochs_for_classification_list = [15]
        ## choose number of epochs to train the the classifier
        #### Recommendation: [15]
        #### Alternative: [5, 10]

        max_length_of_document_vector_w2v_list = [100]
        ## choose the maximum length of the document vector, i.e. the max. number of words of any document in the corpus to include (truncated after this number is reached)
        #### Recommendation: [100]
        #### Alternative: [80, 150]

        classifier_loss_function_w2v_list = ['sparse_categorical_crossentropy']
        ## choose the loss function to use in the Bi LSTM for classification
        #### Recommendation: ['sparse_categorical_crossentropy']
        #### Alternative: ['mean_squared_error', 'sparse_categorical_crossentropy', "kl_divergence", 'categorical_hinge']

        w2v_batch_size_list = [256]  # suggestion: 256
        ## choose the batch-size for training the BiLSTM
        #### Recommendation: [256]
        #### Alternative: [64, 128, 512]

if current_model == "bert":
    small_model_list = [True]
    ## choose whether to use Distilbert or Bert
    #### Recommendation: [True] // use distilbert_uncased (smaller, faster)
    #### Alternative:  [False] // use full bert_uncased (larger, slower)

    bert_batch_size_list = [64]
    ## choose the batch size to train the transformer
    #### Recommendation: [64]
    #### Alternative:  [128, 254] // potentially memory issue on GPUs (depending on hardware)

    bert_epochs_list = [12]
    ## choose for how many epochs to train the transformer model
    #### Recommendation: [12]
    #### Alternative:  [3, 6, 18, 24] // time increases/decreases more or less linearly in the number of epochs

    max_length_of_document_vector_bert_list = [200]
    ## choose maximum number of tokens to from each document to include (choose larger than the number of words because words are split into multiple tokens).
    #### Recommendation: [200] // truncating tokenized documents after 200 tokens
    #### Alternative:  [150, 300]

    classifier_loss_function_bert_list = ['sparse_categorical_crossentropy']
    ## choose the loss function to use in the Bi LSTM for classification
    #### Recommendation: ['sparse_categorical_crossentropy']
    #### Alternative: ['mean_squared_error', 'sparse_categorical_crossentropy', "kl_divergence", 'categorical_hinge']

    use_bert_feature_matrix = False
    ## choose whether to load a pre-existing bert-feature matrix (saved under "data_path/[input_file_name]_[max_length_of_document_vector_bert]_bert_feature_matrix.csv")
    #### Recommendation: False
    #### Alternative: True (load from "data_path/[input_file_name]_[max_length_of_document_vector_bert]_bert_feature_matrix.csv")

    save_bert_feature_matrix = False
    ## choose whether to save thebert-feature matrix (save to "data_path/[input_file_name]_[max_length_of_document_vector_bert]_bert_feature_matrix.csv")
    ## --> PREVIOUS VERSIONS ARE OVERWRITTEN!
    #### Recommendation: False
    #### Alternative: True (save to "data_path/[input_file_name]_[max_length_of_document_vector_bert]_bert_feature_matrix.csv")

'''
apply existing model or train new model?
----------------------------------------------------------
'''
use_model = True
## False if results should not be saved, True if they should be saved // OVERWRITES PREVIOUS FILES WITH THE SAME NAME
#### Recommended: True (use existing model on new data) ; False (train new model)
#### Note: if model is loaded: settings from above


'''
location of model file name
----------------------------------------------------------
'''
model_file_name = "3All_Data_TFIDF"
## Load the model from - or save the model to - a file or folder in DATA_PATH/models// OVERWRITES PREVIOUS FILES WITH THE SAME NAME
#### Recommended: False // automatic name is set based on parameter choices
#### Alternative: e.g. "5TOP5_BERT" // Chose file name from which you want to load the model from or save the model to


'''
settings train a new model
'''
if not use_model:
    save_model = True
    ## False if model should not be saved for later use, True if it should be saved // OVERWRITES PREVIOUS FILES WITH THE SAME NAME

    train_on_all = True
    ## Choose whether to train (and test) on the whole dataset --> RESULTS WILL OVERSTATE PERFORMANCE
    #### Recommended: False // only use a part of the data for training and evaluate the results on a test set
    #### Alternative: True // train a final model on all data

    training_set = "oversample"
    ## Choose whether or not to balance the training set
    #### Recommendation: "oversample" // apply random oversampling to the the minority class to balance the training set
    #### "undersample" // apply random undersampling of the majority class
    #### False // no balancing

    if not train_on_all:
        journal_split = False
        ## select whether to apply cross validation by holding out the articles from one journal at a time
        #### Recommended: False // no cross validation - test- and train-set will include articles from all journals
        #### Alternative: True // fit n models (where n is the number of journals) and evaluate each model on the hold-out journal

        if not journal_split:
            use_reproducible_train_test_split = False
            ## Choose whether a reproducible train-test split to train models or create a new split
            #### Recommended: True // for reproducible results based on previous split ; False // for new, random train-test split should

            if use_reproducible_train_test_split:
                train_set_name = "WOS_lee_heterodox_und_samequality_preprocessed_train_9"
                test_set_name = "WOS_lee_heterodox_und_samequality_preprocessed_test_1"
                ## Choose file names of the train/test files (w/out ".csv")

            if not use_reproducible_train_test_split:
                test_size = 0.1
                ## select the fraction of data to hold out as test set
                #### Recommended: 0.1
                #### Alternative: any number between in (0,1]

        if journal_split:
            num_journals = "all"  # 3 #"all"
            ## select the number of journals to use as hold-out journals (i.e. the number of loops to perform)
            #### Recommended: "all" // use each journals at hold-out set once
            #### Alternative: 1,2,3,4,5,... // choose any number k of journals

            journal_list = False
            ## select specific list of journals to use as hold out set (one after another), by name or number
            #### Recommended: False // use each journal once
            #### Alternative: ["Cambridge Journal of Economics", "Journal of Economic Issues"] // choose a list of journals to use as hold out set (one by one)
            #### Alternative [i for i in range(0, 5)] // choose a list of journals to use as hold out set (one by one)
            #### Alternative [1,5,8,66] // choose a list of journals to use as hold out set (one by one)

#######################################