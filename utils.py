"""Helper functions for the app"""
import pandas as pd
import numpy as np
import tensorflow as tf


class CrossValidationFolds(object):
    """Helper class to perform K-Folds Cross validation dataset splitting,
    divides the dataset into K non-overlapping sections and performs
    train/validation splitting with 1 section as the validation and the rest
    as the train set
    """
    def __init__(self, dataset, labels, num_folds, shuffle=True):
        self.dataset = dataset
        self.labels = labels
        self.num_folds = num_folds
        self.current_fold = 0

        # Shuffle Dataset
        if shuffle:
            perm = np.random.permutation(self.dataset.shape[0])
            self.dataset = dataset[perm, :]
            self.labels = labels[perm, :]

    def split(self):
        current = self.current_fold
        size = int(self.dataset.shape[0] / self.num_folds)

        index = np.arange(self.dataset.shape[0])
        lower_bound = index >= current * size
        upper_bound = index < (current + 1) * size
        cv_region = lower_bound * upper_bound

        cv_data = self.dataset[cv_region, :]
        train_data = self.dataset[~cv_region, :]

        cv_labels = self.labels[cv_region, :]
        train_labels = self.labels[~cv_region, :]

        self.current_fold += 1
        return (train_data, train_labels), (cv_data, cv_labels)


def build_feed(model, features, targets, dropout):

    feed_dict = {model.x: features, model.y_: targets, model.keep_prob: dropout}
    return feed_dict


def one_hot(category, categories_dict):
    """ Encode categoricals into one_hot vectors"""
    one_hot_vector = np.zeros((1, len(categories_dict)), dtype='float32')
    idx = categories_dict[category[0]]
    one_hot_vector.flat[idx] = 1

    return one_hot_vector


def weight_variable(shape, name=None):
    """Creates and initializes trainable tensorflow low-level API Weight matrix"""
    init = tf.truncated_normal(shape=shape, stddev=0.1)
    weight = tf.Variable(init, name=name)
    return weight


def bias_variable(shape, name=None):
    """Creates and initializes trainable tensorflow low-level API bias vector"""
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init, name=name)


def r2(targets, predictions):
    ss_tot = tf.reduce_sum(tf.square(tf.subtract(targets, tf.reduce_mean(targets))))
    ss_res = tf.reduce_sum(tf.square(tf.subtract(targets, predictions)))

    return tf.subtract(1, tf.divide(ss_res, ss_tot))


def preprocess(train_dataframe, test_dataframe, drop_features):
    num_examples = train_dataframe.shape[0]

    # Extracting targets
    target = train_dataframe['y'].values
    target = target.reshape(num_examples, 1)
    del train_dataframe['y']

    # Extracting ID Columns
    test_id = test_dataframe['ID'].values.reshape(num_examples)
    del train_dataframe['ID']
    del test_dataframe['ID']

    # Delete constant features
    for feature in drop_features:
        del train_dataframe[feature]
        del test_dataframe[feature]
        # print('Dropped Feature {}'.format(feature))

    # Categorical and binary features
    categoricals = train_dataframe.columns[train_dataframe.dtypes == object]
    binaries = train_dataframe.columns[train_dataframe.dtypes == 'int64']

    # **Encode categoricals into one_hot vectors**
    categoricals_train = np.empty((num_examples, 0))
    categoricals_test = np.empty((num_examples, 0))

    # Done in a feature by feature basis
    for feature in categoricals:
        union = pd.Series(train_dataframe[feature].tolist() + test_dataframe[feature].tolist()).unique()
        union.sort()

        # Construct dict of categories in feaure
        feature_dict = {}
        for i in range(len(union)):
            feature_dict[union[i]] = i

        # Create one_hot accumulator
        train_one_hot = np.empty((0, len(union)))
        test_one_hot = np.empty((0, len(union)))

        # Create one_hot for each feature separetely, not a vectorized implementation and somewhat obscure
        for i in range(train_dataframe.shape[0]):
            train_one_hot = np.concatenate((train_one_hot, one_hot(train_dataframe[feature].values, feature_dict)))
            test_one_hot = np.concatenate((test_one_hot, one_hot(test_dataframe[feature].values, feature_dict)))

        # Concatenate one_hot of each features into one_hot of all categoricals
        categoricals_train = np.concatenate((categoricals_train, train_one_hot), axis=1)
        categoricals_test = np.concatenate((categoricals_test, test_one_hot), axis=1)

    # concatenate one_hot categoricals and binaries into a full input dataset
    train_data = np.concatenate((categoricals_train, train_dataframe[binaries].values.astype('float32')), axis=1)
    test_input = np.concatenate((categoricals_test, test_dataframe[binaries].values.astype('float32')), axis=1)

    return (train_data, target), (test_input, test_id)
