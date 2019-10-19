import datetime
import os
import pickle
import scipy
from os.path import join

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split

from processing.db_csv import _Dataset
from settings import CSV_CUTTED, JSON_TEXTS, XY_CACHE_FOLDER, NX_SUBGRAPH_PATH


class _DatasetOneUserModel(_Dataset):

    def __init__(self, csv_path=CSV_CUTTED, txt_path=JSON_TEXTS, delta_minutes_filter=None):
        super().__init__(csv_path=csv_path, txt_path=txt_path, delta_minutes_filter=delta_minutes_filter)
        self.left_out_own_retweets_ids = None

    def load_nx_subgraph(self):
        print('Loading graph from {}'.format(NX_SUBGRAPH_PATH))
        return nx.read_gpickle(NX_SUBGRAPH_PATH)

    def _load_df(self, central_uid=None):
        print('Loading df')
        dtypes = {
            'user__id_str': str,
            'id_str': str,
            'retweeted_status__id_str': str,
            'retweeted_status__user__id_str': str,
            'retweet_count': str,
            'quoted_status_id_str': str,
        }
        df = pd.read_csv(self.csv_path, dtype=dtypes)
        original_shape = df.shape

        # parse dates
        datetime_cols = [c for c in df.columns if 'created_at' in c]
        for c in datetime_cols:
            df[c] = pd.to_datetime(df[c])

        # reemplazar nombre de columnas: . por __ para sintactic sugar de pandas.
        df.rename(columns=lambda x: x.replace('.', '__'), inplace=True)
        df.drop_duplicates(subset='id_str', inplace=True)

        if self.delta_minutes:
            print('Filtering by time')
            df_filtered = df[np.isnat(df.retweeted_status__created_at) |
                (df.created_at - df.retweeted_status__created_at <= datetime.timedelta(minutes=self.delta_minutes))]
            df = df_filtered.copy()

            # remove positive cases retweets ids that were filtered in the previous step,
            # because they should not be in the tweets universe (X rows)
            if central_uid is None:
                raise Exception('central_uid must be provided when filtering on time window with one user model')
            left_out_own_retweets = df[(df.user__id_str == central_uid) & pd.notna(df.retweeted_status__id_str) &
                (df.created_at - df.retweeted_status__created_at > datetime.timedelta(minutes=self.delta_minutes))]
            df.drop(left_out_own_retweets.index, inplace=True)

            # self.left_out_own_retweets_ids = left_out_own_retweets.loc[:, 'retweeted_status__id_str'].values
            print('There are {} retweets left out from central user timeline'.format(left_out_own_retweets.shape[0]))

        self.df = df
        print('Done loading df. DF shape is :{} (Original: {}) \t\tTime delta is: {} mins'. \
              format(df.shape, original_shape, self.delta_minutes))

        # self._load_text_df()
        return df

    def get_level2_neighbours(self, user_id):
        """
        An ordered list of up to level-2 neighbours is created
        (followed and their followed)
        """
        g = self.load_nx_subgraph()
        uid = str(user_id)

        neighbourhood = set(list(g.successors(uid)))

        for nid in g.successors(uid):
            neighbourhood.update(g.successors(nid))

        # Remove None elements and own user (TODO: see why this happens)
        neighbour_users = [u_id for u_id in neighbourhood if u_id and u_id != user_id]

        print('Fetched {} level 2 neighbourhs for user {}.'.format(len(neighbour_users), user_id))
        return neighbour_users

    def get_tweets_universe(self, uid, neighbours):
        """Override by child classes. Returns all tweets to be considered for training.
        That is: uid's retweets, plus neighbours tweets in timeline.
        All this pruned to 10000"""
        print('Getting neighbour tweets universe.')
        own_tweets = self.get_user_timeline(uid, with_original=False, with_retweets=True)
        own_tweets_len = own_tweets.shape[0]
        print('Len of own retweets timeline (possible positive examples) is {}'.format(own_tweets_len))

        n_tweets = np.empty((0, 2))
        for u in neighbours:
            # sacar del posible universo de tweets los `left_out_own_tweets`.
            # tl_filtered = tl[np.isin(tl[:, 0], self.left_out_own_retweets_ids, invert=True)]
            n_tweets = np.concatenate((n_tweets, self.get_user_timeline(u)))

        print('Done getting neighbour tweets universe. Shape is ', n_tweets.shape)
        # sacar del posible universo de tweets los `left_out_own_tweets`.
        # n_tweets = n_tweets[np.isin(n_tweets[:, 0], self.left_out_own_retweets_ids, invert=True)]

        # prune to max of 10000. random sample is done over all neighbour tweets.
        n_tweets_len = 10000 - own_tweets_len
        if len(n_tweets) > n_tweets_len:
            idxs = np.random.choice(len(n_tweets), n_tweets_len, replace=False)
            n_tweets = n_tweets[idxs]
            print('\tDataset was truncated to 10000 tweets')

        tweets = np.empty((0, 2))
        tweets = np.concatenate((tweets, own_tweets, n_tweets))

        return tweets

    def get_neighbourhood(self, uid):
        """override with child classes.
        For a given user, returns all users to be used as features in retweeting action"""
        neighbours = self.get_level2_neighbours(uid)
        # remove central user from neighbours
        neighbours = [u for u in neighbours if u != uid]

        return neighbours

    def _extract_features_full(self, tweets, neighbour_users, own_user):
        """
        Given tweets and neighbour_users, we extract
        'neighbour activity' features for each tweet

        These are obtained as follows:
            - for each of these users a boolean feature is created
            indicating if the tweet is authored/retweeted by that user
        """
        nrows = tweets.shape[0]
        nfeats = len(neighbour_users)
        start = datetime.datetime.now()
        X = scipy.sparse.lil_matrix((nrows, nfeats))
        #         X = np.empty((nrows, nfeats), dtype=np.int8)
        #         print('X SIZE (MB): ', X.nbytes  * 1000000)
        y = np.empty(nrows)
        print('Extracting features Optimized. X shape is :', X.shape)

        to_process = tweets.shape[0]
        for j, u in enumerate(neighbour_users):
            to_process -= 1
            percentage = 100.0 - ((to_process / tweets.shape[0]) * 100)
            # print('Avance: %{}'.format(percentage), end='\r')

            n_tl_filtered = self.get_user_timeline(u)
            col = np.isin(tweets[:, 0], n_tl_filtered[:, 0])
            #             print(X[:, j].shape, col.reshape((11,1)))
            X[:, j] = col.reshape((nrows, 1))
            # print(X[:, j])

        own_tl_filtered = self.get_user_timeline(own_user, with_original=False, with_retweets=True)
        y = np.isin(tweets[:, 0], own_tl_filtered[:, 0])
        end = datetime.datetime.now() - start
        print('Done Extracting Features', end)

        # x_sum = sum(sum(X))
        # y_sum = sum(y)
        # print('\tSum of X :{}\n\tSum of y: {}'.format(x_sum, y_sum))
        # if sum(sum(X)) == 0:
        #     raise Exception("Zero matrix X")
        # if sum(y) == 0:
        #     raise Exception("Zero matrix y")

        return X, y

    def extract_features(self, tweets, neighbour_users, own_user, timedelta=None):
        """
        Given tweets and neighbour_users, we extract
        'neighbour activity' features for each tweet

        These are obtained as follows:
            - for each of these users a boolean feature is created
            indicating if the tweet is authored/retweeted by that user and is comprehended in timedelta window
        """
        if not timedelta:
            return self._extract_features_full(tweets, neighbour_users, own_user)

    def load_or_create_dataset(self, uid, delta_minutes_filter):
        self.delta_minutes = delta_minutes_filter
        self.df = pd.DataFrame()
        model_name = self.__class__.__name__
        fname = join(XY_CACHE_FOLDER, "dataset_{}_{}_{}.pickle".format(model_name, uid, self.delta_minutes))
        if os.path.exists(fname):
            dataset = pickle.load(open(fname, 'rb'))
            print('LOADED DATASET FROM {fname}'.format(fname))
        else:
            self._load_df(central_uid=uid)
            uid = str(uid)
            neighbours = self.get_neighbourhood(uid)
            # remove selected user from neighbours
            neighbours = [u for u in neighbours if u != uid]

            tweets = self.get_tweets_universe(uid, neighbours)

            X, y = self.extract_features(tweets, neighbours, uid)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.66666, random_state=42,
                                                                stratify=y_test)
            dataset = (X_train, X_test, X_valid, y_train, y_test, y_valid)

            pickle.dump(dataset, open(fname, 'wb'))

        (X_train, X_test, X_valid, y_train, y_test, y_valid) = dataset
        return dataset


DatasetOneUserModel = _DatasetOneUserModel()
