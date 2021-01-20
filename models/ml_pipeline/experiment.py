import argparse

from sklearn.model_selection import train_test_split

from load_data import load_images
from feature_extractors import extract_hog, extract_frangi

from ml_regressor import MLRegressor

import numpy as np


# -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -=

parser = argparse.ArgumentParser(description='ML Pipeline')

#Arguments
#features
parser.add_argument('--hog', default=False, action='store_true',
                    help='extract Histogram of Oriented Gradients as features (default: True)')
parser.add_argument('--frangi', default=True, action='store_true',
                    help='extract frangi filtering as features (default: True)')

#seed
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')


#folders
parser.add_argument('--data_folder', type=str, default='/Users/falkolavitt/Python/CNN-regressor/data/', metavar='DF',
                    help='path to folder where the data is located (default: /working_directory/data/')

args = parser.parse_args()

def run(args):

    # Directory for saving

    # LOAD DATA ============================================================================================
    print('Loading data...')

    #loading data
    folder = args.data_folder
    train_x, train_y = load_images(folder+'train/')
    test_x, test_y = load_images(folder+'test/')

    #extract features
    print('Extracting features...')
    if args.hog:
        train_x = extract_hog(train_x)
        test_x = extract_hog(test_x)
    if args.frangi:
        train_x = extract_frangi(train_x)
        test_x = extract_frangi(test_x)

    # check if images have the right shape
    if not train_x[0].ndim == 1:
        train_x = np.array([np.reshape(i, (-1)) for i in train_x])
        test_x = np.array([np.reshape(i, (-1)) for i in test_x])



    #run models
    print('Conducting experiments')
    regressor = MLRegressor()
    regressor.run_all_tests()


if __name__ == '__main__':
    run(args)