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
from settings import CSV_CUTTED, JSON_TEXTS, XY_CACHE_FOLDER, NX_SUBGRAPH_PATH, TW_UNIVERSE_CACHE_FOLDER, \
    XY_CACHE_FOLDER_FT


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

        # if self.delta_minutes:
        #     print('Filtering by time')
        #     # df_filtered = df[np.isnat(df.retweeted_status__created_at) |
        #     #     (df.created_at - df.retweeted_status__created_at <= datetime.timedelta(minutes=self.delta_minutes))]
        #     # df = df_filtered.copy()
        #     result_df = df.copy(deep=True)
        #
        #     # remove positive cases retweets ids that were filtered in the previous step,
        #     # because they should not be in the tweets universe (X rows)
        #     if central_uid is None:
        #         raise Exception('central_uid must be provided when filtering on time window with one user model')
        #     left_out_own_retweets = df[(df.user__id_str == central_uid) & pd.notna(df.retweeted_status__id_str) &
        #         (df.created_at - df.retweeted_status__created_at > datetime.timedelta(minutes=self.delta_minutes))]
        #     result_df.drop(left_out_own_retweets.index, inplace=True)  ## wtf emanuel?: VER QUE VAN A FALLAR LOS INDEX ACA
        #
        #     # self.left_out_own_retweets_ids = left_out_own_retweets.loc[:, 'retweeted_status__id_str'].values
        #     print('There are {} retweets left out from central user timeline'.format(left_out_own_retweets.shape[0]))

        self.df = df

        if self.as_seconds:
            print('Done loading df. DF shape is :{} (Original: {}) \t\tTime delta is: {} seconds'. \
                  format(df.shape, original_shape, self.delta_minutes))
        else:
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

    def get_tweets_universe(self, uid, neighbours, try_to_load=True):
        """Override by child classes. Returns all tweets to be considered for training.
        That is: uid's retweets, plus neighbours tweets in timeline.
        All this pruned to 10000"""
        filename = TW_UNIVERSE_CACHE_FOLDER + str(uid)
        if try_to_load:
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    print('Successful load on tweets universe from file: ', filename)
                    return pickle.load(f)

        print('Getting neighbour tweets universe.')
        own_tweets = self.get_user_timeline(uid, with_original=True, with_retweets=True)
        own_tweets_len = own_tweets.shape[0]
        print('Len of own retweets timeline (possible positive examples) is {}'.format(own_tweets_len))

        n_tweets = np.empty((0, 3))
        for u in neighbours:
            # sacar del posible universo de tweets los `left_out_own_tweets`.
            # tl_filtered = tl[np.isin(tl[:, 0], self.left_out_own_retweets_ids, invert=True)]
            n_tweets = np.concatenate((n_tweets, self.get_user_timeline(u,
                                                                        filter_timedelta=False)))

        print('Done getting neighbour tweets universe. Shape is ', n_tweets.shape)
        # sacar del posible universo de tweets los `left_out_own_tweets`.
        # n_tweets = n_tweets[np.isin(n_tweets[:, 0], self.left_out_own_retweets_ids, invert=True)]

        # prune to max of 10000. random sample is done over all neighbour tweets.
        n_tweets_len = 10000 - own_tweets_len
        if len(n_tweets) > n_tweets_len:
            idxs = np.random.choice(len(n_tweets), n_tweets_len, replace=False)
            n_tweets = n_tweets[idxs]
            print('\tDataset was truncated to 10000 tweets')

        tweets = np.empty((0, 3))
        tweets = np.concatenate((tweets, own_tweets, n_tweets))

        with open(filename, 'wb') as f:
            pickle.dump(tweets, f)
        print('Successful save on tweets universe to file: ', filename)

        return tweets

    def get_tweets_universe_2(self, uid, neighbours):  # SPLIT DATASET IN TWO. Train with < timedelta and test with > timedelta
        """Override by child classes. Returns all tweets to be considered for training.
        That is: uid's retweets, plus neighbours tweets in timeline.
        All this pruned to 10000"""
        print('Getting neighbour tweets universe.')
        own_tweets = self.get_user_timeline(uid, with_original=True, with_retweets=True)
        own_tweets_len = own_tweets.shape[0]
        print('Len of own retweets timeline (possible positive examples) is {}'.format(own_tweets_len))

        n_tweets = np.empty((0, 3))
        for u in neighbours:
            # sacar del posible universo de tweets los `left_out_own_tweets`.
            # tl_filtered = tl[np.isin(tl[:, 0], self.left_out_own_retweets_ids, invert=True)]
            n_tweets = np.concatenate((n_tweets, self.get_user_timeline(u,
                                                                        filter_timedelta=True)))  ### TODO: CAPAZ ESTE FILTRO NO VAYA AQUI TAMBIEN

        print('Done getting neighbour tweets universe. Shape is ', n_tweets.shape)
        # sacar del posible universo de tweets los `left_out_own_tweets`.
        # n_tweets = n_tweets[np.isin(n_tweets[:, 0], self.left_out_own_retweets_ids, invert=True)]

        # prune to max of 10000. random sample is done over all neighbour tweets.
        n_tweets_len = 10000 - own_tweets_len
        if len(n_tweets) > n_tweets_len:
            idxs = np.random.choice(len(n_tweets), n_tweets_len, replace=False)
            n_tweets = n_tweets[idxs]
            print('\tDataset was truncated to 10000 tweets')

        tweets = np.empty((0, 3))
        tweets = np.concatenate((tweets, own_tweets, n_tweets))
        # tweets = pd.DataFrame(tweets).drop_duplicates(subset=0)  # drop duplicates on id (by default keeps the first occurrence

        # split on time window

        # first on central user timeline
        # fillna on rt_status__created_at (not retweets) with same value as created_at
        mask = pd.isna(own_tweets[:, 2])
        own_tweets[:, 2] = np.where(mask, own_tweets[:, 1], own_tweets[:, 2])

        mask_positive_early = (own_tweets[:, 1] - own_tweets[:, 2]) < datetime.timedelta(minutes=self.delta_minutes)
        mask_positive_later = ~mask_positive_early

        positive_early = own_tweets[mask_positive_early]
        positive_later = own_tweets[mask_positive_later]

        if positive_later.shape[0] == 0:
            raise Exception('No hay tweets de clase positiva LATER para predecir!')
        if positive_early.shape[0] == 0:
            raise Exception('No hay tweets de clase positiva EARLY para predecir!')

        # second, on neighbour users

        # remove all own_tweets that could be on n_tweets
        n_tweets = n_tweets[~ np.isin(n_tweets[:, 0], own_tweets[:, 0])]

        n_mask_positive_early = (n_tweets[:, 1] - n_tweets[:, 2]) < datetime.timedelta(minutes=self.delta_minutes)
        n_mask_positive_later = ~n_mask_positive_early

        negative_early = n_tweets[n_mask_positive_early]
        negative_later = n_tweets[n_mask_positive_later]

        if negative_later.shape[0] == 0:
            raise Exception('No hay tweets de clase negativa LATER para predecir!')
        if negative_early.shape[0] == 0:
            raise Exception('No hay tweets de clase negativa EARLY para predecir!')

        early_tweets = np.empty((0, 3))
        early_tweets = np.concatenate((early_tweets, positive_early, negative_early))  # SHUFFLE THIS?

        later_tweets = np.empty((0, 3))
        later_tweets = np.concatenate((later_tweets, positive_later, negative_later))  # SHUFFLE THIS?

        return early_tweets, later_tweets

    def get_neighbourhood(self, uid):
        """override with child classes.
        For a given user, returns all users to be used as features in retweeting action"""
        neighbours = self.get_level2_neighbours(uid)
        # remove central user from neighbours
        neighbours = [u for u in neighbours if u != uid]

        return neighbours

    def _extract_features_full(self, tweets, neighbour_users, own_user, fasttext=False):
        """
        Given tweets and neighbour_users, we extract
        'neighbour activity' features for each tweet

        These are obtained as follows:
            - for each of these users a boolean feature is created
            indicating if the tweet is authored/retweeted by that user
        """
        nrows = tweets.shape[0]
        nfeats = len(neighbour_users)
        if fasttext:
            nfeats += 300

        start = datetime.datetime.now()
        X = scipy.sparse.lil_matrix((nrows, nfeats))
        #         X = np.empty((nrows, nfeats), dtype=np.int8)
        #         print('X SIZE (MB): ', X.nbytes  * 1000000)
        y = np.empty(nrows)
        print('Extracting features Optimized. X shape is :', X.shape)

        to_process = tweets.shape[0]
        offset_columns = 0

        if fasttext:
            embeddings = self._get_embeddings_for_tweet(tweets[:, 0])
            for idx, tweet in enumerate(tweets[:, 0]):
                X[idx, :300] = embeddings[idx].split(" ")
            offset_columns = 301

        for j, u in enumerate(neighbour_users, start=offset_columns):
            to_process -= 1
            percentage = 100.0 - ((to_process / tweets.shape[0]) * 100)
            # print('Avance: %{}'.format(percentage), end='\r')

            n_tl_filtered = self.get_user_timeline(u, filter_timedelta=True)
            col = np.isin(tweets[:, 0], n_tl_filtered[:, 0])
            #             print(X[:, j].shape, col.reshape((11,1)))
            X[:, j] = col.reshape((nrows, 1))
            # print(X[:, j])

        own_tl_filtered = self.get_user_timeline(own_user, with_original=True, with_retweets=True)
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

    def extract_features(self, tweets, neighbour_users, own_user, timedelta=None, fasttext=False):
        """
        Given tweets and neighbour_users, we extract
        'neighbour activity' features for each tweet

        These are obtained as follows:
            - for each of these users a boolean feature is created
            indicating if the tweet is authored/retweeted by that user and is comprehended in timedelta window
        """
        if not timedelta:
            return self._extract_features_full(tweets, neighbour_users, own_user, fasttext=fasttext)

    def _add_fasttext_features_to_ds(self, fname_no_ft):
        print('Adding fasttext features to dataset : {}'.format(fname_no_ft))
        dataset = pickle.load(open(fname_no_ft, 'rb'))
        (X_train, X_test, X_valid, y_train, y_test, y_valid, X_train_l, X_test_l, X_valid_l) = dataset
        dataset = list(dataset)
        i = 0
        for ds, labels in zip([X_train, X_test, X_valid], [X_train_l, X_test_l, X_valid_l]):
            nrows = ds.shape[0]
            nfeats = 300

            embeddings_arr = np.empty((nrows, nfeats))

            embeddings = self._get_embeddings_for_tweet(labels)
            for idx, tweet in enumerate(labels):
                embeddings_arr[idx, :] = embeddings[idx].split(" ")
            new_ds = np.c_[embeddings_arr, ds.todense()]
            dataset[i] = new_ds
            i += 1

        return dataset

    def load_or_create_dataset(self, uid, delta_minutes_filter, fasttext=False, as_seconds=False):
        self.as_seconds = as_seconds
        self.delta_minutes = delta_minutes_filter
        # self.df = pd.DataFrame()
        model_name = self.__class__.__name__
        folder = XY_CACHE_FOLDER_FT if fasttext else XY_CACHE_FOLDER
        delta_minutes_fn = str(self.delta_minutes) + 'secs' if as_seconds else str(self.delta_minutes)
        fname = join(folder, "dataset_{}_{}_{}.pickle".format(model_name, uid, delta_minutes_fn))
        fname_no_ft = join(XY_CACHE_FOLDER, "dataset_{}_{}_{}.pickle".format(model_name, uid, delta_minutes_fn))
        print('Intentando cargar dataset social desde: {}'.format(fname))
        if os.path.exists(fname):
            dataset = pickle.load(open(fname, 'rb'))
            print('LOADED DATASET FROM {fname}'.format(fname=fname))
        elif fasttext and os.path.exists(fname_no_ft):
            print('Cargado dataset social con exito')
            dataset = self._add_fasttext_features_to_ds(fname_no_ft)
        else:
            raise Exception('NO ERA POR ACA ...')
            if self.df.empty:
                print('DF EMPTY LOADING IT')
                self._load_df(central_uid=uid)
            uid = str(uid)
            neighbours = self.get_neighbourhood(uid)
            # remove selected user from neighbours
            neighbours = [u for u in neighbours if u != uid]

            tweets = self.get_tweets_universe(uid, neighbours)

            X, y = self.extract_features(tweets, neighbours, uid, fasttext=fasttext)
            labels = tweets[:, 0]

            X_train, X_test, y_train, y_test, X_train_l, X_test_l = train_test_split(X, y, labels, test_size=0.3,
                                                                                     random_state=42, stratify=y)
            X_valid, X_test, y_valid, y_test, X_valid_l, X_test_l = train_test_split(X_test, y_test, X_test_l, test_size=0.66666, random_state=42,
                                                                stratify=y_test)
            dataset = (X_train, X_test, X_valid, y_train, y_test, y_valid, X_train_l, X_test_l, X_valid_l)

            pickle.dump(dataset, open(fname, 'wb'))

            (X_train, X_test, X_valid, y_train, y_test, y_valid, X_train_l, X_test_l, X_valid_l) = dataset
        return dataset

    @staticmethod
    def get_matrix_density(minutes, as_seconds=False):
        from processing.utils import get_test_users_ids
        uids = get_test_users_ids()
        ds = []
        ds_r = []
        for uid in uids:
            timewindow = '{}{}'.format(minutes, 'secs' if as_seconds else '')
            filename = 'processing/xy_cache/dataset__DatasetOneUserModel_{}_{}.pickle'.format(uid, timewindow)
            with open(filename, 'rb') as f:
                X_train, X_valid, X_testv, y_train, y_valid, y_testv, X_train_l, X_test_l, X_valid_l = pickle.load(f)
            X_t = X_train.todense()
            row_with_most_ones = np.sum(X_t[X_t.sum(axis=1).argmax()])

            matrix_size = 1.0 * X_t.shape[0] * X_t.shape[1]
            matrix_rows = 1.0 * X_t.shape[0]
            matrix_ones = np.sum(X_t)
            density = matrix_ones / matrix_size
            density_r = matrix_ones / matrix_rows
            ds.append(density)
            ds_r.append(density_r)
        from statistics import mean
        mean_cell_ds = mean(ds)
        mean_row_ds = mean(ds_r)
        print('Mean cell density (size / ones): {}'.format(mean_cell_ds))
        print('Mean row density (size / ones): {}'.format(mean_row_ds))

    # a = []
    # for n in nei:
    #     DatasetOneUserModel.delta_minutes = 2
    #
    #     if '1029021872862711808' in DatasetOneUserModel.get_user_timeline(n, filter_timedelta=True)[:, 0]:
    #         a.append(n)

    # def load_or_create_dataset_2(self, uid, delta_minutes_filter):
    #     """
    #     Implementation that splits train-test according to time window.
    #     :param uid:
    #     :param delta_minutes_filter:
    #     :return:
    #     """
    #     self.delta_minutes = delta_minutes_filter
    #     self.df = pd.DataFrame()
    #     model_name = self.__class__.__name__
    #     fname = join(XY_CACHE_FOLDER, "dataset_{}_{}_{}.pickle".format(model_name, uid, self.delta_minutes))
    #     if os.path.exists(fname):
    #         dataset = pickle.load(open(fname, 'rb'))
    #         print('LOADED DATASET FROM {fname}'.format(fname=fname))
    #     else:
    #         self._load_df(central_uid=uid)
    #         uid = str(uid)
    #         neighbours = self.get_neighbourhood(uid)
    #         # remove selected user from neighbours
    #         neighbours = [u for u in neighbours if u != uid]
    #
    #         tweets_train, tweets_test = self.get_tweets_universe_2(uid, neighbours)
    #         print('\t\tTrain tweets shape is: {}; test tweets shape is: {}'.format(tweets_train.shape, tweets_test.shape))
    #
    #         X_train, y_train = self.extract_features(tweets_train, neighbours, uid)
    #         X_test, y_test = self.extract_features(tweets_test, neighbours, uid)
    #         X_valid, y_valid = None, None
    #
    #         # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    #         # X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.66666, random_state=42,
    #         #                                                     stratify=y_test)
    #         dataset = (X_train, X_test, X_valid, y_train, y_test, y_valid)
    #
    #         pickle.dump(dataset, open(fname, 'wb'))
    #
    #     (X_train, X_test, X_valid, y_train, y_test, y_valid) = dataset
    #     return dataset


DatasetOneUserModel = _DatasetOneUserModel()
