import os

from utils import AttributeDict

GLOBAL = {
    'VERBOSE': os.environ.get('GLOBAL_VERBOSE', False)
}

VERBOSE = False

DATASETS_FOLDER = '../../datasets/'

FIRST_SAMPLE = DATASETS_FOLDER + 'twitter_sample.db'

SQLITE_CONNECTION = FIRST_SAMPLE
