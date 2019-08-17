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
JSON_TEXTS = None

if DATASET_SIZE_TYPE == 'SMALL':
    SQLITE_CONNECTION = 'sqlite:///../data/processed/twitter_sample_daily.db'
    CSV_RAW = '../data/raw/csvs/dayli.csv'
    CSV_CUTTED = '../data/raw/csvs/cut1_dayli.csv'
    JSON_TEXTS = '../data/raw/jsons/texts_dayli.json'

if DATASET_SIZE_TYPE == 'FULL':
    SQLITE_CONNECTION = 'sqlite:///../data/processed/twitter_sample_full.db'
    CSV_RAW = '../data/raw/csvs/full.csv'
    CSV_CUTTED = '../data/raw/csvs/cut1_full.csv'
    JSON_TEXTS = '../data/raw/jsons/texts_full.json'

print("""SQLITE_CONNECTION = {}
CSV_RAW = {}
CSV_CUTTED = {}
JSON_TEXTS = {}""".format(SQLITE_CONNECTION, CSV_RAW, CSV_CUTTED, JSON_TEXTS))

XY_CACHE_FOLDER = "./processing/xy_cache/"

NX_GRAPH_PATH = '../data/graphs/subgraph_new_downloaded.gpickle'

NX_SUBGRAPH_PATH = '../data/graphs/latest_subgraph.gpickle'

NX_GRAPH_FOLDER = '../data/graphs/'

ACTIVE_AND_CENTRAL = "../data/processed/active_and_central.pickle"  # 206 users

MODELS_FOLDER_1_ = "../model/_1_one_user_learn_neighbours"

SCORES_FOLDER_1_ =  "modeling/_1_one_user_learn_neighbours/scores"

CENTRAL_USER_DATA = {
    'id': 393285785, ## NOT IN NEW_DOWNLOADED.gpickle!!!!!!!!!!!
    'screen_name': 'PCelayes'
}