import json
import os


# GLOBAL = {
#     'VERBOSE': os.environ.get('GLOBAL_VERBOSE', False)
# }

VERBOSE = False

DATASETS_FOLDER_RAW = '../data/processed/'

DATASET_SIZE_TYPE = os.environ.get('DATASET_SIZE_TYPE')
SQLITE_CONNECTION = None
CSV_RAW = None
CSV_CUTTED = None

if DATASET_SIZE_TYPE == 'SMALL':
    SQLITE_CONNECTION = 'sqlite:///../data/processed/twitter_sample_daily.db'
    CSV_RAW = '../data/raw/csvs/dayli_col.csv'
    CSV_CUTTED = '../data/raw/csvs/cut1_dayli_col.csv'
    print("""SQLITE_CONNECTION = 'sqlite:///../data/processed/twitter_sample_daily.db'
    CSV_RAW = '../data/raw/csvs/dayli_col.csv'
    CSV_CUTTED = '../data/raw/csvs/cut1_dayli_col.csv'""")

if DATASET_SIZE_TYPE == 'FULL':
    SQLITE_CONNECTION = 'sqlite:///../data/processed/twitter_sample_daily.db'
    CSV_RAW = '../data/raw/csvs/full.csv'
    CSV_CUTTED = '../data/raw/csvs/cut1_full_col.csv'
    print("""SQLITE_CONNECTION = 'sqlite:///../data/processed/twitter_sample_daily.db'
    CSV_RAW = '../data/raw/csvs/full.csv'
    CSV_CUTTED = '../data/raw/csvs/cut1_full_col.csv'""")

XY_CACHE_FOLDER = "./processing/xy_cache/"

GT_GRAPH_PATH = '../data/graphs/subgraph.gt'

NX_GRAPH_PATH = '../data/graphs/subgraph_new_downloaded.gpickle'

tu_path = "../data/processed/active_and_central.json"
TEST_USERS_ALL = json.load(open(tu_path))

MODELS_FOLDER_1_ = "../model/_1_one_user_learn_neighbours"

SCORES_FOLDER_1_ =  "modeling/_1_one_user_learn_neighbours/scores"
