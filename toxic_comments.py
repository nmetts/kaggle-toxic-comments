"""
A module to transform features, cross-validate, or make predictions for the Toxic Comments contest.

Created on January 18, 2018

@author: Nicolas Metts
"""

import argparse
import csv
import os
import re
import sys

import numpy as np
import pandas as pd
from mlxtend.classifier import StackingClassifier
from nltk.stem import WordNetLemmatizer
from scipy.sparse import csc_matrix, hstack
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from textblob import TextBlob

# Name of label columns
LABEL_COLUMNS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Actions
CREATE_FEATURE_FILES = 'create_feature_files'
CROSS_VALIDATE = 'cross_validate'
PREDICT_TEST = 'predict_test'

# Features
LEMMATIZE = 'lemmatize'
SENTIMENT = 'sentiment'
CORRECT_SPELLING = 'correct_spelling'
POS_REPLACE = 'pos_replace'
REMOVE_PUNCTUATION = 'remove_punctuation'
SPECIAL_CHARACTER_COUNT = 'special_character_count'
NUM_WORDS = 'num_words'
MEAN_WORD_LENGTH = 'mean_word_length'
UNIQUE_WORDS = 'num_unique_words'

ADDITIONAL_COLUMN_FEATURES = [SPECIAL_CHARACTER_COUNT, NUM_WORDS, MEAN_WORD_LENGTH, UNIQUE_WORDS]

# Classifiers
RANDOM_FOREST = 'random_forest'
LOGISTIC_REGRESSION = 'logistic_regression'
EXTRA_TREES = 'extra_trees'
MLP = 'mlp'
STACKING = 'stacking'

row_index = 0


def correct_spelling(row_str):
    return str(TextBlob(row_str).correct())


def pos_replace(row_str):
    tags = TextBlob(row_str).tags
    tag_replace = ['CC', 'CD', 'DT', 'IN', 'NN', 'NNP', 'NNS', 'PRP', 'PRP$', 'WP']
    return " ".join([x[1] if x[1] in tag_replace else x[0] for x in tags])


def get_sentiment_analysis(row_str):
    blob = TextBlob(row_str)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


def get_polarity(row):
    return row[0]


def get_subjectivity(row):
    return row[1]


def get_num_words(row_str):
    return len(row_str.split())


def get_mean_word_length(row_str):
    if len(row_str.split()) == 0:
        return 0
    else:
        return len(row_str.replace(" ", "")) / len(row_str.split())


def get_unique_words(row_str):
    return len(set(row_str.split()))


def get_special_character_count(row_str):
    return len(re.sub(r"\w", "", row_str.replace(" ", "")))


def remove_punctuation(row_str):
    try:
        global row_index
        row_index += 1
        return re.sub(r"\W", " ", row_str)
    except TypeError:
        print("Type error on: {}".format(row_str))
        print("Type is: {}".format(type(row_str)))
        print("Row index: {}".format(row_index))
        exit(1)


def lemmatize(row_str):
    wnl = WordNetLemmatizer()
    return " ".join([wnl.lemmatize(x) for x in row_str.split()])


def get_features(df, features):
    if not features:
        return df
    if REMOVE_PUNCTUATION in features:
        df = df.assign(comment_text=df.comment_text.apply(remove_punctuation))
    if CORRECT_SPELLING in features:
        print("Correcting spelling")
        df = df.assign(comment_text=df.comment_text.apply(correct_spelling))
    if LEMMATIZE in features:
        print("Lemmatizing words")
        df = df.assign(comment_text=df.comment_text.apply(lemmatize))
    if SENTIMENT in features:
        print("Performing sentiment analysis")
        t = df.comment_text.apply(get_sentiment_analysis)
        df = df.assign(polarity=t.apply(get_polarity))
        df = df.assign(subjectivity=t.apply(get_subjectivity))
    if SPECIAL_CHARACTER_COUNT in features:
        print("Finding special character count")
        df = df.assign(special_character_count=df.comment_text.apply(get_special_character_count))
    if NUM_WORDS in features:
        print("Finding number of words")
        df = df.assign(num_words=df.comment_text.apply(get_num_words))
    if MEAN_WORD_LENGTH in features:
        print("Finding mean word length")
        df = df.assign(mean_word_length=df.comment_text.apply(get_mean_word_length))
    if UNIQUE_WORDS in features:
        print("Finding number of unique words")
        df = df.assign(num_unique_words=df.comment_text.apply(get_unique_words))
    if POS_REPLACE in features:
        print("Replacing words with POS tag")
        df = df.assign(comment_text=df.comment_text.apply(pos_replace))
    return df


def create_feature_files(train_data, test_data, features):
    """
    A function to create CSV files that can be used to directly predict or cross-validate,
    thus eliminating the need to do feature engineering.

    Params:
        train_data(str): The name of the train CSV file
        test_data(str): The name of the test CSV file
        param features(list): A list of features to be used
    """
    train_df = pd.read_csv(train_data)
    train_labels = train_df[LABEL_COLUMNS]

    # Save the train labels to file if not already saved
    label_file_name = "{}{}train_labels.csv".format(os.path.dirname(train_data), os.path.sep)
    if not os.path.exists(label_file_name):
        train_labels.to_csv(label_file_name, index=False, index_label=False)

    train_df.drop(LABEL_COLUMNS, axis=1, inplace=True)
    test_df = pd.read_csv(test_data, na_filter=False)
    test_ids = test_df.id.values

    # Save the test ids if not already saved
    id_file_name = "{}{}test_ids.txt".format(os.path.dirname(test_data), os.path.sep)
    if not os.path.exists(id_file_name):
        with open(id_file_name, 'w') as id_file:
            for test_id in test_ids:
                id_file.write("{}\n".format(test_id))

    print("Getting features for train data")
    train_features = get_features(train_df, features)
    print("Getting features for test data")
    test_features = get_features(test_df, features)

    vectorizer = TfidfVectorizer()
    print("Fitting and transforming train data")
    train_matrix = vectorizer.fit_transform(train_features.comment_text)
    print("Transforming test data")
    test_matrix = vectorizer.transform(test_features.comment_text)

    if 'subjectivity' in train_features.columns and 'polarity' in train_features.columns:
        print("Adding polarity and subjectivity to train matrix")
        train_matrix = hstack((train_matrix, train_features[['subjectivity', 'polarity']]))

        print("Adding polarity and subjectivity to test matrix")
        test_matrix = hstack((test_matrix, test_features[['subjectivity', 'polarity']]))

    additional_columns = [x for x in train_features.columns if x in ADDITIONAL_COLUMN_FEATURES]
    train_matrix = hstack((train_matrix, train_features[additional_columns]), format='csc')
    test_matrix = hstack((test_matrix, test_features[additional_columns]), format='csc')

    train_path = os.path.dirname(train_data)
    test_path = os.path.dirname(test_data)
    new_train_name = "{}_{}{}{}.csv".format(train_path, os.path.sep,
                                            os.path.basename(train_data).split('.csv')[0],
                                            "_".join(features))
    new_test_name = "{}_{}{}{}.csv".format(test_path, os.path.sep,
                                           os.path.basename(test_data).split('.csv')[0],
                                           "_".join(features))

    print("Writing train matrix to file")
    np.savez(new_train_name, data=train_matrix.data, indices=train_matrix.indices,
             indptr=train_matrix.indptr, shape=train_matrix.shape)
    print("Writing test matrix to file")
    np.savez(new_test_name, data=test_matrix.data, indices=test_matrix.indices,
             indptr=test_matrix.indptr, shape=test_matrix.shape)


def get_classifiers(clf_names):
    """
    A function to get a list of classifiers.

    Params:
        clf_names(list): A list of classifier names

    Return:
         A list of classifier objects.
    """
    clf_list = []

    if LOGISTIC_REGRESSION in clf_names:
        clf_list.append(LogisticRegressionCV(n_jobs=-1, solver='newton-cg', scoring=log_loss, penalty='l2',
                                             class_weight='balanced', multi_class='multinomial'))
    if RANDOM_FOREST in clf_names:
        clf_list.append(RandomForestClassifier(n_jobs=-1, n_estimators=400, class_weight='balanced'))
    if MLP in clf_names:
        clf_list.append(MLPClassifier(hidden_layer_sizes=(500, 200, 100)))
    if EXTRA_TREES in clf_names:
        clf_list.append(ExtraTreesClassifier(n_jobs=-1, n_estimators=400, class_weight='balanced'))
    if STACKING in clf_names:
        # See if we can avoid hitting the recursion limit
        sys.setrecursionlimit(2000)
        meta_clf = RandomForestClassifier(n_jobs=-1, n_estimators=400, class_weight='balanced')
        clf = StackingClassifier(classifiers=clf_list, meta_classifier=meta_clf, use_probas=True)
        clf_list.append(clf)
    return clf_list


def predict(train_file, labels_file, test_file, id_file, classifiers):
    """
    A function to make predictions and write them to file. All parameters are required.

    Params:
        train_file(str): The name of the train file
        labels_file(str): The name of the labels file
        test_file(str): The name of the test file
        classifiers(list): A list of classifier names
    """
    loader = np.load(train_file)
    train_data = csc_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
    loader = np.load(test_file)
    test_data = csc_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
    labels_mat = pd.read_csv(labels_file).as_matrix()
    ids = np.loadtxt(id_file)

    clf_list = get_classifiers(classifiers)

    for clf in clf_list:
        print("Using classifier: {}".format(clf))
        print("Fitting to train data")
        clf.fit(X=train_data, y=labels_mat)
        print("Making predictions")
        predictions = np.array(clf.predict_proba(test_data))
        p_shape = predictions.shape
        predictions = predictions.reshape(p_shape[1], p_shape[0], p_shape[2])
        predictions = predictions[:, :, 1]
        preds_file_name = os.path.dirname(test_file) + os.path.sep + clf.__class__.__name__ + "_predictions.csv"

        print("Writing predictions to file")
        with open(preds_file_name, 'w') as preds_file:
            writer = csv.writer(preds_file)
            header_row = ['id'] + LABEL_COLUMNS
            writer.writerow(header_row)
            for row_id, preds in zip(ids, predictions):
                row = [row_id] + list(preds)
                writer.writerow(row)


def get_mean_log_loss(y_true, y_pred):
    """
    A function to calculate the mean log loss across columns

    Params:
        y_true(numpy.array):  The Numpy array of true labels
        y_pred(numpy.array): The Numpy array of log probabilities for each label column

    Throws AssertionError if y_true.shape != y_pred.shape
    """
    assert (y_true.shape == y_pred.shape)
    return np.mean([log_loss(y_true=y_true[..., col_idx], y_pred=y_pred[..., col_idx])
                    for col_idx in range(y_true.shape[1])])


def cross_validate(train_file, labels_file, classifiers):
    """
    A function to perform cross-validation

    Params:
        train_file(str): The name of the train file. Expected to be a CSV file transformed and ready for fitting.
        labels_file(str): The name of the file with labels
        classifiers(list): A list of classifiers to be used in cross-validation
    """
    loader = np.load(train_file)
    data = csc_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
    labels_mat = pd.read_csv(labels_file).as_matrix()
    data = hstack((data, labels_mat))

    train_data, test_data = train_test_split(data, test_size=0.2)
    # Separate the labels columns from the train and test features
    train_labels = (train_data[:, -6:]).toarray()
    test_labels = (test_data[:, -6:]).toarray()
    train_data = train_data[:, :-6]
    test_data = test_data[:, :-6]
    clf_list = get_classifiers(classifiers)

    for clf in clf_list:
        print("Using classifier: {}".format(clf.__class__.__name__))
        print("Fitting to train data")
        clf.fit(X=train_data, y=train_labels)
        print("Making predictions")
        predictions = np.array(clf.predict_proba(test_data))
        p_shape = predictions.shape
        if len(p_shape) == 3:
            predictions = predictions.reshape(p_shape[1], p_shape[0], p_shape[2])
            predictions = predictions[:, :, 1]
        print("Calculating log loss")
        loss = get_mean_log_loss(y_true=test_labels, y_pred=predictions)
        print("Loss is: {}".format(loss))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train_file', type=str, help='Name of the train data', required=True)
    argparser.add_argument('--test_file', type=str, help='Name of the test data file')
    argparser.add_argument('--labels_file', type=str, help='Name of the labels file. Required for cross-validation')
    argparser.add_argument('--id_file', type=str, help='Name of the id file. Required for predictions')
    argparser.add_argument('--features', nargs='+', help='The features to be used')
    argparser.add_argument('--classifiers', nargs='+', help='The features to be used')
    argparser.add_argument('--action', type=str, help='Name of the action to take',
                           choices=[CREATE_FEATURE_FILES, CROSS_VALIDATE, PREDICT_TEST])
    args = argparser.parse_args()

    if args.action == CROSS_VALIDATE:
        cross_validate(train_file=args.train_file, labels_file=args.labels_file, classifiers=args.classifiers)
    elif args.action == CREATE_FEATURE_FILES:
        create_feature_files(train_data=args.train_file, test_data=args.test_file, features=args.features)
    elif args.action == PREDICT_TEST:
        predict(train_file=args.train_file, test_file=args.test_file, labels_file=args.labels_file,
                id_file=args.id_file, classifiers=args.classifiers)
