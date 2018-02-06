"""
A module to transform features, cross-validate, or make predictions for the Toxic Comments contest.

Created on January 18, 2018

@author: Nicolas Metts
"""

import argparse
import csv
import datetime as dt
import multiprocessing
import os
import re
import sys

import numpy as np
import pandas as pd
from mlxtend.classifier import StackingClassifier
from nltk.stem import WordNetLemmatizer
from scipy.sparse import csc_matrix, hstack
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob

# Name of label columns
LABEL_COLUMNS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# File types
DF = 'df'
NPZ = 'npz'

# Actions
CREATE_FEATURE_FILES = 'create_feature_files'
CROSS_VALIDATE = 'cross_validate'
PREDICT_TEST = 'predict_test'

# Features
TF_IDF = 'tf_idf'
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
CLASSIFIER_CHAIN = 'classifier_chain'
MULTIOUTPUT_CLASSIFIER = 'multioutput_classifier'
EXTRA_TREES = 'extra_trees'
MLP = 'mlp'
STACKING = 'stacking'

MLP_CLASSIFIER_NAME = 'MLPClassifier'

# Functions to be applied to DataFrame columns


def correct_spelling(row_str):
    """
    A function to correct the spelling of a document.

    Params:
        row_str(str): The document to be corrected.
    """
    return str(TextBlob(row_str).correct())


def pos_replace(row_str):
    """
    A function to replace the words in a document with their part of speech (POS) tag.

    Params:
        row_str(str): The document to be transformed
    """
    tags = TextBlob(row_str).tags
    tag_replace = ['CC', 'CD', 'DT', 'IN', 'NN', 'NNP', 'NNS', 'PRP', 'PRP$', 'WP']
    return " ".join([x[1] if x[1] in tag_replace else x[0] for x in tags])


def get_sentiment_analysis(row_str):
    """
    A function to obtain the sentiment analysis (polarity and subjectivity) of a document.

    Params:
        row_str(str): The document for which to obtain sentiment analysis.
    """
    blob = TextBlob(row_str)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


def get_polarity(row):
    """
    Get the polarity from the tuple of (polarity, subjectivity) resulting from sentiment analysis.

    Params:
        row(tuple): The row containing subjectivity and polarity.
    """
    return row[0]


def get_subjectivity(row):
    """
    Get the subjectivity from the tuple of (subjectivity, polarity) resulting from sentiment analysis.

    Params:
        row(tuple): The row containing subjectivity and polarity.
    """
    return row[1]


def get_num_words(row_str):
    """
    A function to obtain the number of words in a document.

    Params:
        row_str(str): The document
    """
    return len(row_str.split())


def get_mean_word_length(row_str):
    """
    A function to obtain the mean word length of a document.

    Params:
        row_str(str): The document
    """
    if len(row_str.split()) == 0:
        return 0
    else:
        return len(row_str.replace(" ", "")) / len(row_str.split())


def get_unique_words(row_str):
    """
    A function to obtain the number of unique words in a document.

    Params;
        row_str(str): The document
    """
    return len(set(row_str.split()))


def get_special_character_count(row_str):
    """
    A function to get the number of special characters in a document.
    Params:
    row_str(str): The document
    """
    return len(re.sub(r"\w", "", row_str.replace(" ", "")))


def remove_punctuation(row_str):
    """
    A function to remove the punctuation from a document
    Params:
        row_str(str): The document to be transformed.
    """
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
    """
    A function to lemmatize the words in a document.

    Params:
        row_str(str): The document to be lemmatized
    """
    wnl = WordNetLemmatizer()
    return " ".join([wnl.lemmatize(x) for x in row_str.split()])


def get_features(df, features):
    """
    A function to obtain all the features when creating a feature file.

    Params:
        df(pandas.DataFrame): The DataFrame to be used in feature generation.
        features(list): A list of features to be obtained
    """
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


def persist_model(clf):
    """
    A convenience function for persisting a trained model to disk.

    Params:
        clf(sklearn.base.BaseEstimator): A trained (fitted) estimator
    """
    dt_str = dt.datetime.now().strftime('%m_%d_%Y_%H')
    print("Saving model: {}".format(clf.__class__.__name__))
    pickle_file_name = '{}_{}.pkl'.format(clf.__class__.__name__, dt_str)
    joblib.dump(clf, pickle_file_name)


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

    train_path = os.path.dirname(train_data)
    test_path = os.path.dirname(test_data)
    new_train_name = "{}{}{}_{}.csv".format(train_path, os.path.sep,
                                            os.path.basename(train_data).split('.csv')[0],
                                            "_".join(features))
    new_test_name = "{}{}{}_{}.csv".format(test_path, os.path.sep,
                                           os.path.basename(test_data).split('.csv')[0],
                                           "_".join(features))

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

    if TF_IDF in features:
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

        # Remove the .csv extension if saving in npz format
        new_train_name = new_train_name.split(".csv")[0]
        new_test_name = new_test_name.split(".csv")[0]
        print("Writing train matrix to file")
        np.savez(new_train_name, data=train_matrix.data, indices=train_matrix.indices,
                 indptr=train_matrix.indptr, shape=train_matrix.shape)
        print("Writing test matrix to file")
        np.savez(new_test_name, data=test_matrix.data, indices=test_matrix.indices,
                 indptr=test_matrix.indptr, shape=test_matrix.shape)
    else:
        train_features.drop(['comment_text', 'id'], axis=1, inplace=True)
        test_features.drop(['comment_text', 'id'], axis=1, inplace=True)
        train_features.to_csv(new_train_name, index=False, index_label=False)
        test_features.to_csv(new_test_name, index=False, index_label=False)


def get_classifiers(clf_names):
    """
    A function to get a list of classifiers.

    Params:
        clf_names(list): A list of classifier names

    Return:
         A list of classifier objects.
    """
    clf_list = []

    if RANDOM_FOREST in clf_names:
        clf_list.append(RandomForestClassifier(n_jobs=-1, n_estimators=400, class_weight='balanced'))
    if MLP in clf_names:
        clf_list.append(MLPClassifier(hidden_layer_sizes=(200, 100, 50), verbose=True, max_iter=1000))
    if EXTRA_TREES in clf_names:
        clf_list.append(ExtraTreesClassifier(n_jobs=-1, n_estimators=400, class_weight='balanced'))
    if CLASSIFIER_CHAIN in clf_names:
        if clf_list:
            base_estimator = clf_list[0]
        else:
            num_cpus = multiprocessing.cpu_count()
            base_estimator = SGDClassifier(loss='log', n_jobs=int(num_cpus/6) - 1,
                                           tol=1e-5, max_iter=1000, class_weight='balanced')
        clf_list.append(ClassifierChain(base_estimator=base_estimator))
    if MULTIOUTPUT_CLASSIFIER in clf_names:
        if clf_list:
            base_estimator = clf_list[0]
        else:
            num_cpus = multiprocessing.cpu_count()
            base_estimator = SGDClassifier(loss='log', n_jobs=int(num_cpus/6) - 1,
                                           tol=1e-5, max_iter=1000, class_weight='balanced')
        clf_list.append(MultiOutputClassifier(estimator=base_estimator, n_jobs=6))
    if STACKING in clf_names:
        # See if we can avoid hitting the recursion limit
        sys.setrecursionlimit(2000)
        meta_clf = RandomForestClassifier(n_jobs=-1,  class_weight='balanced')
        clf = StackingClassifier(classifiers=clf_list, meta_classifier=meta_clf, use_probas=True)
        clf_list.append(clf)
    return clf_list


def write_predictions_to_file(preds_file_name, ids, predictions):
    """
    A convenience function for writing predictions to file.

    Params:
        preds_file_name(str): The name of the predictions file.
        ids(list or numpy.array): An iterable containing prediction ids
        predictions(numpy.array): An array of shape (num_predictions, num_classes) containing predictions
    """
    print("Writing predictions to file")
    with open(preds_file_name, 'w') as preds_file:
        writer = csv.writer(preds_file)
        header_row = ['id'] + LABEL_COLUMNS
        writer.writerow(header_row)
        for row_id, preds in zip(ids, predictions):
            row = [row_id] + list(preds)
            writer.writerow(row)


def get_predictions(clf, test_data):
    """
    Convenience function for getting predictions.

    Params:
        clf(sklearn.BaseEstimator): A fitted (trained) estimator used to make predictions
        test_data(np.array or scipy.sparse.csc_matrix): The test data used for predictions
    """
    print("Making predictions")
    predictions = np.array(clf.predict(test_data))
    p_shape = predictions.shape
    if len(p_shape) == 3:
        predictions = predictions.reshape(p_shape[1], p_shape[0], p_shape[2])
        predictions = predictions[:, :, 1]
    return predictions


def predict(train_file, labels_file, test_file, id_file, file_type,
            classifiers, save_model=False, use_model=None, scale=False):
    """
    A function to make predictions and write them to file. If using a saved model, the train_file is not required,
    and neither is labels_file.

    Params:
        train_file(str): The name of the train file
        labels_file(str): The name of the labels file
        test_file(str): The name of the test file
        id_file(str): The name of a file containing ids for making predictions
        file_type(str): Indicates the file type. Either Pandas DataFrame or scipy.sparse.csc_matrix
        classifiers(list): A list of classifier names
        save_model(bool): Indicates whether the trained model should be persisted to disk.
        use_model(bool): Indicates whether a pre-trained model should be used. If so, classifiers is not used.
    """
    if use_model:
        clf = joblib.load(use_model)
        loader = np.load(test_file)
        test_data = csc_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
        ids = pd.read_csv(id_file).id.values
        predictions = get_predictions(clf=clf, test_data=test_data)
        preds_file_name = os.path.dirname(test_file) + os.path.sep + clf.__class__.__name__ + "_predictions.csv"
        write_predictions_to_file(preds_file_name=preds_file_name, ids=ids, predictions=predictions)

    else:
        ids = pd.read_csv(id_file).id.values
        if file_type == NPZ:
            loader = np.load(train_file)
            train_data = csc_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
            loader = np.load(test_file)
            test_data = csc_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
            labels_mat = pd.read_csv(labels_file).as_matrix()

        else:
            train_data = pd.read_csv(train_file)
            test_data = pd.read_csv(test_file)
            labels_mat = pd.read_csv(labels_file).as_matrix()

        clf_list = get_classifiers(classifiers)

        for clf in clf_list:
            print("Using classifier: {}".format(clf))
            if (clf.__class__.__name__ == MLP_CLASSIFIER_NAME) or scale:
                print("Fitting scaler")
                scaler = StandardScaler()
                scaler.fit(train_data)
                print("Transforming train data")
                train_data = scaler.transform(train_data)
                print("Transforming test data")
                test_data = scaler.transform(test_data)
            print("Fitting to train data")
            clf.fit(train_data, labels_mat)
            if save_model:
                persist_model(clf)
            predictions = get_predictions(clf=clf, test_data=test_data)
            preds_file_name = os.path.dirname(test_file) + os.path.sep + clf.__class__.__name__ + "_predictions.csv"
            write_predictions_to_file(preds_file_name=preds_file_name, ids=ids, predictions=predictions)


def get_mean_log_loss(y_true, y_pred):
    """
    A function to calculate the mean log loss across columns.

    Params:
        y_true(numpy.array):  The Numpy array of true labels
        y_pred(numpy.array): The Numpy array of log probabilities for each label column

    Throws AssertionError if y_true.shape != y_pred.shape
    """
    p_shape = y_pred.shape

    # Need to reshape predictions array for some classifiers.
    if len(p_shape) == 3:
        y_pred = y_pred.reshape(p_shape[1], p_shape[0], p_shape[2])
        y_pred = y_pred[:, :, 1]
    assert (y_true.shape == y_pred.shape)
    return np.mean([log_loss(y_true=y_true[..., col_idx], y_pred=y_pred[..., col_idx])
                    for col_idx in range(y_true.shape[1])])


def cross_validate(train_file, labels_file, file_type, classifiers, save_model, scale=False):
    """
    A function to perform cross-validation

    Params:
        train_file(str): The name of the train file. Expected to be a CSV file transformed and ready for fitting.
        labels_file(str): The name of the file with labels
        file_type(str): Indicates the file type. Either Pandas DataFrame or scipy.sparse.csc_matrix
        classifiers(list): A list of classifiers to be used in cross-validation
        save_model(bool): Indicates whether the fitted model should be saved to disk.
    """
    if file_type == NPZ:
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
    else:
        data = pd.read_csv(train_file)
        labels = pd.read_csv(labels_file)
        data = data.merge(labels, right_index=True, left_index=True)
        train_data, test_data = train_test_split(data, test_size=0.2)

        # Separate the labels columns from the train and test features
        train_labels = (train_data[LABEL_COLUMNS]).as_matrix()
        test_labels = (test_data[LABEL_COLUMNS]).as_matrix()
        train_data.drop(LABEL_COLUMNS, axis=1, inplace=True)
        test_data.drop(LABEL_COLUMNS, axis=1, inplace=True)

    clf_list = get_classifiers(classifiers)

    for clf in clf_list:
        print("Using classifier: {}".format(clf.__class__.__name__))
        print("Fitting to train data")
        if (clf.__class__.__name__ == MLP_CLASSIFIER_NAME) or scale:
            print("Fitting scaler")
            scaler = StandardScaler()
            scaler.fit(train_data)
            print("Transforming train data")
            train_data = scaler.transform(train_data)
            print("Transforming test data")
            test_data = scaler.transform(test_data)
        clf.fit(train_data, train_labels)
        if save_model:
            persist_model(clf=clf)
        print("Making predictions")
        predictions = np.array(clf.predict(test_data))
        p_shape = predictions.shape
        if len(p_shape) == 3:
            predictions = predictions.reshape(p_shape[1], p_shape[0], p_shape[2])
            predictions = predictions[:, :, 1]
        assert (test_labels.shape == predictions.shape)
        print("Calculating ROC AUC score")
        loss = roc_auc_score(y_true=test_labels, y_score=predictions)
        print("ROC AUC is: {}".format(loss))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train_file', help='Name of the train data', required=True)
    argparser.add_argument('--test_file', help='Name of the test data file')
    argparser.add_argument('--labels_file', help='Name of the labels file. Required for cross-validation')
    argparser.add_argument('--id_file', help='Name of the id file. Required for predictions')
    argparser.add_argument('--file_type', help='Type of file for train and test data.',
                           choices=[DF, NPZ], default=NPZ)
    argparser.add_argument('--features', nargs='+', help='The features to be used')
    argparser.add_argument('--classifiers', nargs='+', help='The features to be used')
    argparser.add_argument('--save_model', action='store_true', help='Whether the model should be saved')
    argparser.add_argument('--use_model', help='The path to a saved model')
    argparser.add_argument('--scale', action='store_true', help='Whether data should be scaled')
    argparser.add_argument('--action', help='Name of the action to take',
                           choices=[CREATE_FEATURE_FILES, CROSS_VALIDATE, PREDICT_TEST])
    args = argparser.parse_args()

    if args.action == CROSS_VALIDATE:
        cross_validate(train_file=args.train_file, labels_file=args.labels_file, file_type=args.file_type,
                       classifiers=args.classifiers, save_model=args.save_model, scale=args.scale)
    elif args.action == CREATE_FEATURE_FILES:
        create_feature_files(train_data=args.train_file, test_data=args.test_file, features=args.features)
    elif args.action == PREDICT_TEST:
        predict(train_file=args.train_file, test_file=args.test_file, labels_file=args.labels_file,
                id_file=args.id_file, file_type=args.file_type, classifiers=args.classifiers,
                save_model=args.save_model, use_model=args.use_model, scale=args.scale)
