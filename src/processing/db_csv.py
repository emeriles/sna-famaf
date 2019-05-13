import os
import pickle
import datetime
from os.path import join

import pandas as pd
import numpy as np
import scipy
from sklearn.model_selection import train_test_split

from processing.utils import load_nx_graph
from settings import CSV_CUTTED, XY_CACHE_FOLDER


class _Dataset(object):

    def __init__(self, csv_path=CSV_CUTTED, delta_minutes_filter=None):
        self.csv_path = csv_path
        self.df: pd.DataFrame = pd.DataFrame()
        self.delta_minutes = delta_minutes_filter

    def get_most_active_users(self, N=1000, just_ids=True):
        most_active = sorted(self.df.id_str.groupby(self.df.user__id_str).count().iteritems(),
                             reverse=True, key=lambda x: x[1])
        if just_ids:
            return [id_ for id_, counts in most_active][:N]
        return most_active[:N]

    def _load_df(self):
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
            df_filtered = df[
                (df.created_at - df.retweeted_status__created_at <= datetime.timedelta(minutes=240)) |
                np.isnat(df.retweeted_status__created_at)]
            df = df_filtered.copy()

        self.df = df
        print('Done loading df. DF shape is :{} (Original: {}) \t\tTime delta is: {} mins'. \
              format(df.shape, original_shape, self.delta_minutes))
        return df

    def get_level2_neighbours(self, user_id):
        """
        An ordered list of up to level-2 neighbours is created
        (followed and their followed)
        """
        g = load_nx_graph()
        uid = str(user_id)

        neighbourhood = set(list(g.successors(uid)))

        for nid in g.successors(uid):
            neighbourhood.update(g.successors(nid))

        # Remove None elements and own user (TODO: see why this happens)
        neighbour_users = [u_id for u_id in neighbourhood if u_id and u_id != user_id]
        # print(neighbour_users[5:])

        return neighbour_users

    def get_all_users(self):
        if self.df.empty:
            self._load_df()
        return self.df.user__id_str.unique()

    def get_user_timeline(self, uid):
        """
        Returns [(tweet_id, creted_at)}
        :param uid:
        :return:
        """
        if self.df.empty:
            self._load_df()
        if isinstance(uid, str) or isinstance(uid, int):
            uid = [uid]
        filtered = self.df[(self.df.user__id_str.isin(uid)) | (self.df.retweeted_status__user__id_str.isin(uid))]
        tweets = filtered.copy()
        own_tweets = tweets.loc[:, ('id_str', 'created_at')]
        rts = tweets.loc[:, ('retweeted_status__id_str', 'retweeted_status__created_at')]
        rts.rename({'retweeted_status__id_str': 'id_str',
                    'retweeted_status__created_at': 'created_at'},
                   axis='columns', inplace=True)
        timeline = pd.concat([own_tweets, rts]).dropna().drop_duplicates(subset='id_str').values
        return timeline[:10000]

    def get_tweets_universe(self, uid, neighbours):
        # universe_of_tweets = self.df[
        #                          ~((self.df.user__id_str == uid) & pd.isna(self.df.retweeted_status__id_str))  # no tweets originales de uid
        #                          & (self.df.user__id_str.isin(neighbours) | self.df.retweeted_status__user__id_str.isin(neighbours))  # tweets en neighbours
        #                      ]
        #
        # # join tweets and retweets in same columns
        # tweets = universe_of_tweets.loc[:, ('id_str', 'created_at')]
        # rts = universe_of_tweets.loc[:, ('retweeted_status__id_str', 'retweeted_status__created_at')]
        # rts.rename({'retweeted_status__id_str': 'id_str',
        #             'retweeted_status__created_at': 'created_at'},
        #            axis='columns', inplace=True)
        # result = pd.concat([tweets, rts]).dropna().drop_duplicates(subset='id_str').values
        tweets = np.empty((0,2))
        for u in neighbours:
            tweets = np.concatenate((tweets, self.get_user_timeline(u)))

        print('done getting tweets universe. Shape is ', tweets.shape)
        return tweets
        # return tweets

    # def extract_features(tweets, neighbour_users, own_user):
    #     '''
    #         Given tweets and neighbour_users, we extract
    #         'neighbour activity' features for each tweet
    #
    #         These are obtained as follows:
    #             - for each of these users a boolean feature is created
    #             indicating if the tweet is authored/retweeted by that user
    #     '''
    #     nrows = len(tweets)
    #     nfeats = len(neighbour_users)
    #     X = np.empty((nrows, nfeats))
    #     print('X SIZE (MB): ', X.nbytes  * 1000000)
    #     y = np.empty(nrows)
    #
    #     own_tl_full = [(t.id, t.created_at) for t in own_user.timeline]
    #     for j, u in enumerate(neighbour_users):
    #         tl_full = [(t.id, t.created_at) for t in u.timeline]
    #         for i, t in enumerate(tweets):
    #             # additional filtering on time constraints
    #             tl_ids = [tw.id for (tw, c) in tl_full if c > t.created_at]
    #             X[i, j] = 1 if t.id in tl_ids else 0
    #
    #     for i, t in enumerate(tweets):
    #         # additional filtering on time constraints
    #         own_tl_ids = [tw.id for (tw, c) in own_tl_full if tw.created_at > t.created_at]
    #         y[i] = 1 if t.id in own_tl_ids else 0
    #
    #     return X, y

    # def extract_features(self, tweets, neighbour_users, own_user):
    #     """
    #     Given tweets and neighbour_users, we extract
    #     'neighbour activity' features for each tweet
    #
    #     These are obtained as follows:
    #         - for each of these users a boolean feature is created
    #         indicating if the tweet is authored/retweeted by that user
    #     """
    #     nrows = tweets.shape[0]
    #     nfeats = len(neighbour_users)
    #     start = datetime.datetime.now()
    #     print('Extracting features. X shape is: ', nrows, nfeats, )
    #     X = np.empty((nrows, nfeats), dtype=np.int8)
    #     print('X SIZE (MB): ', X.nbytes * 1000000)
    #     y = np.empty(nrows)
    #
    #     to_process = tweets.shape[0]
    #     for j, u in enumerate(neighbour_users):
    #         to_process -= 1
    #         percentage = 100.0 - ((to_process / tweets.shape[0]) * 100)
    #         print('Avance: %{}'.format(percentage), end='\r')
    #         tl_full = self.get_user_timeline(u)
    #         for i, t in enumerate(tweets):
    #             # additional filtering on time constraints
    #             # tl_ids = [tw for (tw, c) in tl_full if c > t[1]]  # TIME DELTA
    #             # n_tl_filtered = self.get_user_timeline(u)  # FILTER HERE
    #
    #             X[i, j] = np.isin(t[0], tl_full[:, 0])
    #             # tl_ids = [tw for (tw, c) in tl_full if c > t[1]]  # TIME DELTA
    #             # X[i, j] = 1 if t[0] in tl_ids else 0
    #
    #     own_tl_full = self.get_user_timeline(own_user)
    #     for i, t in enumerate(tweets):
    #         # additional filtering on time constraints
    #         own_tl_filtered = own_tl_full  # FILTER HERE
    #         y[i] = np.isin(t[0], own_tl_filtered[:, 0])
    #         # own_tl_ids = [tw for (tw, c) in own_tl_full if tw[1] > t[1]]
    #         # y[i] = 1 if t[0] in own_tl_ids else 0
    #     end = datetime.datetime.now() - start
    #     print('Done Extracting Features', end)
    #     return X, y

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
            print('Avance: %{}'.format(percentage), end='\r')

            n_tl_filtered = self.get_user_timeline(u)
            col = np.isin(tweets[:, 0], n_tl_filtered[:, 0])
            #             print(X[:, j].shape, col.reshape((11,1)))
            X[:, j] = col.reshape((nrows, 1))
            # print(X[:, j])

        own_tl_filtered = self.get_user_timeline(own_user)
        y = np.isin(tweets[:, 0], own_tl_filtered[:, 0])
        end = datetime.datetime.now() - start
        print('Done Extracting Features', end)
        return X, y

    def extract_features(self, tweets, neighbour_users, own_user, timedelta=None):
        """
        Given tweets and neighbour_users, we extract
        'neighbour activity' features for each tweet

        These are obtained as follows:
            - for each of these users a boolean feature is created
            indicating if the tweet is authored/retweeted by that user
        """
        if not timedelta:
            return self._extract_features_full(tweets, neighbour_users, own_user)
    # nrows = tweets.shape[0]
    # nfeats = len(neighbour_users)
    # start = datetime.datetime.now()
    # X = np.empty((nrows, nfeats), dtype=np.int8)
    # print('X SIZE (MB): ', X.nbytes  * 1000000)
    # y = np.empty(nrows)
    # print('Extracting features. X shape is :', X.shape)
    #
    # to_process = tweets.shape[0]
    # # RARO: neighbour_users son 5173
    # t_delta = np.timedelta64(timedelta, 'm')
    # plus_time_delta = (tweets[:, 1].astype("datetime64[m]") + t_delta)
    # for j, u in enumerate(neighbour_users):
    #     to_process -= 1
    #     percentage = 100.0 - (to_process / tweets.shape[0]) * 100
    #     print('Avance: {}'.format(percentage), end='\r')
    #
    #     # traigo timeline del vecino
    #     n_tl = self.get_user_timeline(u)
    #     # selecciono solo los tweets de n_tl que estan en `tweets`
    #     tweets_found_on_tl = n_tl[np.isin(n_tl[:, 0], tweets[:, 0])]
    #     # sacar los indices que corresponden con la lista de tweets
    #     idx_with_rt = np.where(np.isin(tweets[:, 0], n_tl[:, 0]))[0]
    #     # create empty column for X
    #     x_col_datetime = np.full_like(tweets[:, 1], np.datetime64('nat'), dtype='datetime64[m]')
    #     # fill x_col_datetime on idx_with_rt with values from tweets_found_on_tl
    #     # import ipdb; ipdb.set_trace()
    #     x_col_datetime[idx_with_rt] = tweets_found_on_tl[:, 1]
    #     # x_col_datetime = x_col_datetime.astype("datetime64[m]") + t_delta
    #     X[:, j] = (plus_time_delta - x_col_datetime) >= np.timedelta64(0, 'm')
    #
    # own_tl_filtered = self.get_user_timeline(own_user)
    # y = np.isin(tweets[:, 0], own_tl_filtered[:, 0])
    # end = datetime.datetime.now() - start
    # print('Done Extracting Features', end)
    # return X, y

    def get_neighbourhood(self, uid):
        neighbours = self.get_level2_neighbours(uid)
        # remove central user from neighbours
        neighbours = [u for u in neighbours if u != uid]

        return neighbours

    def load_or_create_dataset(self, uid, delta_minutes_filter):
        self.delta_minutes = delta_minutes_filter
        fname = join(XY_CACHE_FOLDER, "dataset_{}_{}.pickle".format(uid, self.delta_minutes))
        if os.path.exists(fname):
            dataset = pickle.load(open(fname, 'rb'))
        else:
            self._load_df()
            uid = str(uid)
            neighbours = self.get_level2_neighbours(uid)
            # remove central user from neighbours
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

    # def load_or_create_dataset_validation(self, uid):
    #     print('Load_or_create_dataframe_validation for user: ', uid)
    #     result = self.load_validation_dataset(uid)
    #     if not result:
    #         print('Creating dataframe for user: ', uid)
    #         self.repartition_dataframe(uid)  # se guarda x_valid
    #         X_train, X_valid, X_test, y_train, y_valid, y_test = self.reduce_dataset(uid)  # se guardan los x_valid_small
    #         return X_train, X_valid, X_test, y_train, y_valid, y_test
    #     print('Returning loaded model.')
    #     return result

    # def reduce_dataset(self, uid):
    #     ds = self.load_or_create_dataset(uid)
    #     X_train, X_test, y_train, y_test = ds
    #
    #     X = np.concatenate((X_train,X_test))
    #     y = np.concatenate((y_train,y_test))
    #
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #     X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.66666, random_state=42)
    #
    #     Xtrain_fname = join(XY_CACHE_FOLDER, "Xtrain_%d_small.pickle" % uid)
    #     Xvalid_fname = join(XY_CACHE_FOLDER, "Xvalid_%d_small.pickle" % uid)
    #     Xtest_fname = join(XY_CACHE_FOLDER, "Xtestv_%d_small.pickle" % uid)
    #     ys_fname = join(XY_CACHE_FOLDER, "ysv_%d_small.pickle" % uid)
    #
    #     np.save(Xtrain_fname, X_train)
    #     np.save(Xvalid_fname, X_valid)
    #     np.save(Xtest_fname, X_test)
    #     pickle.dump((y_train, y_valid, y_test), open(ys_fname, 'wb'))
    #
    #     return X_train, X_valid, X_test, y_train, y_valid, y_test

    # def load_validation_dataset(self, uid):
    #     X_train, X_test, X_valid, y_train, y_test, y_valid = self.load_or_create_dataset(uid)
    #     return X_valid, y_valid

    # def load_dataframe(self, uid):
    #     Xtrain_fname = join(XY_CACHE_FOLDER, "dfXtrain_%d_small.pickle" % uid)
    #     Xvalid_fname = join(XY_CACHE_FOLDER, "dfXvalid_%d_small.pickle" % uid)
    #     Xtest_fname = join(XY_CACHE_FOLDER, "dfXtestv_%d_small.pickle" % uid)
    #     ys_fname = join(XY_CACHE_FOLDER, "ysv_%d_small.pickle" % uid)
    #     try:
    #         X_train = np.load(Xtrain_fname)
    #         X_valid = np.load(Xvalid_fname)
    #         X_test = np.load(Xtest_fname)
    #         y_train, y_valid, y_test = pickle.load(open(ys_fname, 'rb'))
    #         return X_train, X_valid, X_test, y_train, y_valid, y_test
    #     except Exception as e:
    #         return None

    # def repartition_dataframe(self, uid):
    #     ds = self.load_or_create_dataset(uid)  # load_dataframe(uid)
    #
    #     if ds:
    #         X_train, X_test, y_train, y_test = ds
    #         X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test,
    #                                                 test_size=0.6667, random_state=42)
    #
    #         # Xvalid_fname = join(XY_CACHE_FOLDER, "Xvalid_%d.pickle" % uid)
    #         Xvalid_fname = join(XY_CACHE_FOLDER, "Xvalid_%d.npy" % uid)
    #         # Xtest_fname = join(XY_CACHE_FOLDER, "Xtestv_%d.pickle" % uid)
    #         Xtest_fname = join(XY_CACHE_FOLDER, "Xtestv_%d.npy" % uid)
    #         # ys_fname = join(XY_CACHE_FOLDER, "ysv_%d.pickle" % uid)
    #         ys_fname = join(XY_CACHE_FOLDER, "ysv_%d.npy" % uid)
    #
    #         Xtest_fname_old = join(XY_CACHE_FOLDER, "Xtest_%d.npy" % uid)
    #
    #         # X_train.to_pickle(Xtrain_fname)
    #         np.save(Xvalid_fname, X_valid)
    #         np.save(Xtest_fname, X_test)
    #         np.save(Xtest_fname, X_test)
    #         pickle.dump((y_train, y_valid, y_test), open(ys_fname, 'wb'))
    #
    #         try:
    #             remove(Xtest_fname_old)
    #         except FileNotFoundError:
    #             pass

    # def load_small_validation_dataframe(self, uid):
    #     Xtrain_fname = join(XY_CACHE_FOLDER, "dfXtrain_%d_small.pickle" % uid)
    #     Xvalid_fname = join(XY_CACHE_FOLDER, "dfXvalid_%d_small.pickle" % uid)
    #     Xtest_fname = join(XY_CACHE_FOLDER, "dfXtestv_%d_small.pickle" % uid)
    #     ys_fname = join(XY_CACHE_FOLDER, "ysv_%d_small.pickle" % uid)
    #
    #     X_train = pd.read_pickle(Xtrain_fname)
    #     X_valid = pd.read_pickle(Xvalid_fname)
    #     X_test = pd.read_pickle(Xtest_fname)
    #     y_train, y_valid, y_test = pickle.load(open(ys_fname, 'rb'))
    #
    #     return X_train, X_valid, X_test, y_train, y_valid, y_test


Dataset = _Dataset()
