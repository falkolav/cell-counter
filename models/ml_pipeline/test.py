import argparse

from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

from load_data import load_images
from feature_extractors import extract_hog, extract_frangi

from ml_regressor import MLRegressor

parser = argparse.ArgumentParser(description='ML Pipeline')

#Arguments
#features
parser.add_argument('--hog', default=False, action='store_true',
                    help='extract Histogram of Oriented Gradients as features (default: False)')
parser.add_argument('--frangi', default=False, action='store_true',
                    help='extract frangi filtering as features (default: False)')

#seed
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')

args = parser.parse_args()

def run(args):

    # Directory for saving

    # LOAD DATA ============================================================================================
    print('Loading data...')

    #declare variables
    global train_x
    global train_y
    global test_x
    global test_y

    #loading data
    folder = '/Users/falkolavitt/Python/CNN-regressor/data/'
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

    #split train/val/test
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=args.seed)

    #check if images have the right shape
    if not train_x[0].ndim == 1:
        train_x = np.array([np.reshape(i, (-1)) for i in train_x])
        test_x = np.array([np.reshape(i, (-1)) for i in test_x])

    #run models
    print('Conducting experiments')
    regressor = MLRegressor(train_x, train_y, test_x, test_y)
    regressor.run_all_tests()

    #plot results
    plt.boxplot(regressor.results.loc['error'])
    plt.xticks(ticks=[1, 2, 3, 4, 5], labels=['SVR', 'XGB', 'RR', 'NNR', 'GTB'])
    plt.savefig('ml_pipeline_boxplot.png')
    plt.close('all')

if __name__ == '__main__':
    run(args)