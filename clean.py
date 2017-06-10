"""Clean and concise implementation of this neural network. Performs K-folds cross-validation
    may be used for parameter tuning"""

import pandas as pd
import numpy as np
from network import Mercedez, train_model
from utils import preprocess, CrossValidationFolds

# Constants
PATH = './data/'
TRAIN = 'train.csv'
TEST = 'test.csv'
SUBMIT = False

# Hyper Parameters
MAX_ITER = 400
LEARNING_RATE = 1e-3
LAYERS = [567, 500, 250]
FOLDS = 5
DROPOUT = 0.3

FEATURE_DROP = ['X11', 'X93', 'X107', 'X223', 'X235', 'X268', 'X289',
                'X290', 'X293', 'X297', 'X330', 'X347']


def read_data(path, train, test):
    print('Reading CSV Data...')
    train_df = pd.read_csv(path + train)
    test_df = pd.read_csv(path + test)

    num_examples = train_df.shape[0]  # Both the training and test set have the same # of examples
    print('Data Read\n')

    return train_df, test_df, num_examples


def k_folds(model, num_folds, train_features, targets):
    r_squared_log = []
    mse_log = []
    data = CrossValidationFolds(train_features, targets, num_folds)

    for i in range(num_folds):
        print('Current fold: {}\n'.format(data.current_fold + 1))
        (train_input, train_target), (cv_input, cv_target) = data.split()

        # Start Training
        cv_loss, cv_r2 = train_model(data.current_fold, MAX_ITER, DROPOUT,
                                     model, train_input, train_target, cv_input, cv_target)

        mse_log.append(cv_loss)
        r_squared_log.append(cv_r2)

    final_mse = np.array(mse_log).mean()
    final_r_squared = np.array(r_squared_log).mean()
    print('K folds finished')
    print('Final validation score, MSE: {0: .2f} | R_squared: {1: .5f}'.format(final_mse, final_r_squared))


def main():
    # Reading data
    train_df, test_df, num_examples = read_data(PATH, TRAIN, TEST)

    # Preprocessing
    (train_data, target), (_, _) = preprocess(train_df, test_df, FEATURE_DROP)

    # Define model & build graph
    model = Mercedez(batch_size=num_examples, layers=LAYERS, learning_rate=LEARNING_RATE)
    model.build_graph()

    # Run K-Folds Validation
    k_folds(model, FOLDS, train_data, target)

if __name__ == '__main__':
    main()
