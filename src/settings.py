import os
from os.path import dirname, abspath, join


# GLOBAL = {
#     'VERBOSE': os.environ.get('GLOBAL_VERBOSE', False)
# }

VERBOSE = False

DATASETS_FOLDER_RAW = '../data/processed/'

XY_CACHE_FOLDER = "./processing/xy_cache/"

GT_GRAPH_PATH = '../data/graphs/subgraph.gt'

NX_GRAPH_PATH = '../data/graphs/subgraph.gpickle'

SQLITE_CONNECTION = 'sqlite:///../data/processed/twitter_sample.db'

SQLITE_CONNECTION_SMALL = 'sqlite:///../data/processed/twitter_sample_daily.db'

CSV_SMALL = '../data/csvs/dayli_col.csv'

CSV_FULL = '../data/csvs/full.csv'
