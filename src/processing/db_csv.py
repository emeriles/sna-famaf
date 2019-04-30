import os
import pickle
from datetime import datetime
from os.path import join

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from processing.utils import load_nx_graph
from settings import CSV_CUTTED, XY_CACHE_FOLDER


class _Dataset(object):

    def __init__(self, csv_path=CSV_CUTTED):
        self.csv_path = csv_path
        self.df: pd.DataFrame = pd.DataFrame()

    def _load_df(self, parse_dates=True):
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

        # parse dates
        if parse_dates:
            datetime_cols = [c for c in df.columns if 'created_at' in c]
            for c in datetime_cols:
                df[c] = pd.to_datetime(df[c])

        # reemplazar nombre de columnas: . por __ para sintactic sugar de pandas.
        df.rename(columns=lambda x: x.replace('.', '__'), inplace=True)
        self.df = df
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
        own_twets = tweets.loc[:, ('id_str', 'created_at')]
        rts = tweets.loc[:, ('retweeted_status__id_str', 'retweeted_status__created_at')]
        rts.rename({'retweeted_status__id_str': 'id_str',
                    'retweeted_status__created_at': 'created_at'},
                   axis='columns', inplace=True)
        timeline = pd.concat([own_twets, rts]).dropna().drop_duplicates().values
        return timeline

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
    #     y = np.empty(nrows)
    #
        # own_tl_full = [(t.id, t.created_at) for t in own_user.timeline]
        # for j, u in enumerate(neighbour_users):
        #     tl_full = [(t.id, t.created_at) for t in u.timeline]
        #     for i, t in enumerate(tweets):
        #         # additional filtering on time constraints
        #         tl_ids = [tw.id for (tw, c) in tl_full if c > t.created_at]
        #         X[i, j] = 1 if t.id in tl_ids else 0
        #
        # for i, t in enumerate(tweets):
        #     # additional filtering on time constraints
        #     own_tl_ids = [tw.id for (tw, c) in own_tl_full if tw.created_at > t.created_at]
        #     y[i] = 1 if t.id in own_tl_ids else 0
        #
        # return X, y

    # def extract_features(self, tweets, neighbour_users, own_user):
    #     """
    #     Given tweets and neighbour_users, we extract
    #     'neighbour activity' features for each tweet
    #
    #     These are obtained as follows:
    #         - for each of these users a boolean feature is created
    #         indicating if the tweet is authored/retweeted by that user
    #     """
    #     nrows = len(tweets)
    #     nfeats = len(neighbour_users)
    #     start = datetime.now()
    #     print('Extracting features', nrows, nfeats)
    #     X = np.empty((nrows, nfeats), dtype=np.int8)
    #     y = np.empty(nrows)
    #
    #     for j, u in enumerate(neighbour_users):
    #         tl_full = self.get_user_timeline(u)
    #         for i, t in enumerate(tweets):
    #             # additional filtering on time constraints
    #             tl_ids = [tw for (tw, c) in tl_full if c > t[1]]  # TIME DELTA
    #             X[i, j] = 1 if t[0] in tl_ids else 0
    #
    #     own_tl_full = self.get_user_timeline(own_user)
    #     for i, t in enumerate(tweets):
    #         # additional filtering on time constraints
    #         own_tl_ids = [tw for (tw, c) in own_tl_full if tw[1] > t[1]]
    #         y[i] = 1 if t[0] in own_tl_ids else 0
    #     end = datetime.now() - start
    #     print('Done Extracting Features', end)
    #     return X, y


    def extract_features(self, tweets, neighbour_users, own_user):
        """
        Given tweets and neighbour_users, we extract
        'neighbour activity' features for each tweet

        These are obtained as follows:
            - for each of these users a boolean feature is created
            indicating if the tweet is authored/retweeted by that user
        """
        nrows = tweets.shape[0]
        nfeats = len(neighbour_users)
        start = datetime.now()
        print('Extracting features', nrows, nfeats)
        X = np.empty((nrows, nfeats), dtype=np.int8)
        y = np.empty(nrows)

        percentage = 0
        to_process = len(neighbour_users)
        for j, u in enumerate(neighbour_users):
            to_process -= 1
            percentage = 100 - int((to_process / len(neighbour_users)) * 100)
            print(
                'Avance: %{}'.format(percentage), end='\r'
            )
            # additional filtering on time constraints
            n_tl_filtered = self.get_user_timeline(u)  # FILTER ON TIME
            # import ipdb; ipdb.set_trace()
            # print(tweets[:5], n_tl_filtered[:5])
            col = np.isin(tweets[:, 0], n_tl_filtered[:, 0])
            # tl_ids = [tw for (tw, c) in tl_full if c > t[1]]  # TIME DELTA
            X[:, j] = col

        own_tl_filtered = self.get_user_timeline(own_user) # FILTER ON TIME
        y = np.isin(tweets[:, 0], own_tl_filtered[:, 0])
        # y = np.isin
        # for i, t in enumerate(tweets):
        #     # additional filtering on time constraints
        #     own_tl_ids = [tw for (tw, c) in own_tl_full if tw[1] > t[1]]
        #     y[i] = 1 if t[0] in own_tl_ids else 0
        end = datetime.now() - start
        print('Done Extracting Features', end)
        return X, y

    n1 = ['1057277373417168898','1057269914191519744','1057265600756637697','1057226333472915456','1057225175966588929','1057221770976129024','1057096423508979713']
    def get_neighbourhood(self, uid):
        neighbours = self.get_level2_neighbours(uid)
        # remove central user from neighbours
        neighbours = [u for u in neighbours if u != uid]

        return neighbours

    # from former datasets.py
    def load_or_create_dataset(self, uid):
        fname = join(XY_CACHE_FOLDER, "dataset_%d.pickle" % uid)
        if os.path.exists(fname):
            dataset = pickle.load(open(fname, 'rb'))
        else:
            uid = str(uid)
            neighbours = self.get_level2_neighbours(uid)
            # remove central user from neighbours
            neighbours = [u for u in neighbours if u != uid]

            # Fetch tweet universe (timelines of ownuser and neighbours)
            own_tweets = list(self.get_user_timeline(uid))
            print(uid)
            print(uid)
            print(uid)
            print('OWN : {}'.format(own_tweets[:10]))
            n_tweets = list(self.get_user_timeline(neighbours))
            print('N NNNNNN : {}'.format(n_tweets[:10]))
            tweets = [(t, c) for (t, c) in n_tweets if t not in [i for (i,c) in own_tweets]] # n_tweets.difference(own_tweets)  # TODO: add language selection?
            tweets = np.array(tweets)
            # exclude tweets from central user or not in Spanish
            # tweets = [t for t in tweets if t.author_id != uid ]  # and t.lang == 'es'   all tweets in spanish?

            X, y = self.extract_features(tweets, neighbours, uid)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            dataset = (X_train, X_test, y_train, y_test)

            pickle.dump(dataset, open(fname, 'wb'))

        (X_train, X_test, y_train, y_test) = dataset
        print((X_train[:5], y_train[:5]))
        return dataset

    def load_or_create_dataframe_validation(self, uid):
        print('Load_or_create_dataframe_validation for user: ', uid)
        result = self.load_dataframe(uid)
        if not result:
            print('Creating dataframe for user: ', uid)
            self.repartition_dataframe(uid)  # se guarda x_valid
            X_train, X_valid, X_test, y_train, y_valid, y_test = self.reduce_dataset(uid)  # se guardan los x_valid_small
            return X_train, X_valid, X_test, y_train, y_valid, y_test
        print('Returning loaded model.')
        return result

    def load_dataframe(self, uid):
        Xtrain_fname = join(XY_CACHE_FOLDER, "dfXtrain_%d_small.pickle" % uid)
        Xvalid_fname = join(XY_CACHE_FOLDER, "dfXvalid_%d_small.pickle" % uid)
        Xtest_fname = join(XY_CACHE_FOLDER, "dfXtestv_%d_small.pickle" % uid)
        ys_fname = join(XY_CACHE_FOLDER, "ysv_%d_small.pickle" % uid)
        try:
            X_train = pd.read_pickle(Xtrain_fname)
            X_valid = pd.read_pickle(Xvalid_fname)
            X_test = pd.read_pickle(Xtest_fname)
            y_train, y_valid, y_test = pickle.load(open(ys_fname, 'rb'))
            return X_train, X_valid, X_test, y_train, y_valid, y_test
        except Exception as e:
            return None

    def repartition_dataframe(self, uid):
        ds = self.load_or_create_dataframe(uid)  # load_dataframe(uid)

        if ds:
            X_train, X_test, y_train, y_test = ds
            X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test,
                                                    test_size=0.6667, random_state=42)

            Xvalid_fname = join(XY_CACHE_FOLDER, "dfXvalid_%d.pickle" % uid)
            Xtest_fname = join(XY_CACHE_FOLDER, "dfXtestv_%d.pickle" % uid)
            ys_fname = join(XY_CACHE_FOLDER, "ysv_%d.pickle" % uid)

            Xtest_fname_old = join(XY_CACHE_FOLDER, "dfXtest_%d.pickle" % uid)

            # X_train.to_pickle(Xtrain_fname)
            X_valid.to_pickle(Xvalid_fname)
            X_test.to_pickle(Xtest_fname)
            pickle.dump((y_train, y_valid, y_test), open(ys_fname, 'wb'))

            remove(Xtest_fname_old)

    def load_validation_dataframe(self, uid):
        Xtrain_fname = join(XY_CACHE_FOLDER, "dfXtrain_%d.pickle" % uid)
        Xvalid_fname = join(XY_CACHE_FOLDER, "dfXvalid_%d.pickle" % uid)
        Xtest_fname = join(XY_CACHE_FOLDER, "dfXtestv_%d.pickle" % uid)
        ys_fname = join(XY_CACHE_FOLDER, "ysv_%d.pickle" % uid)

        X_train = pd.read_pickle(Xtrain_fname)
        X_valid = pd.read_pickle(Xvalid_fname)
        X_test = pd.read_pickle(Xtest_fname)
        y_train, y_valid, y_test = pickle.load(open(ys_fname, 'rb'))

        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def reduce_dataset(self, uid):
        ds = self.load_validation_dataframe(uid)
        X_train, X_valid, X_test, y_train, y_valid, y_test = ds

        X=pd.concat((X_train,X_valid,X_test))
        y=np.concatenate((y_train,y_valid,y_test))

        if len(y) > 5000:
            neg_inds = [i for i, v in enumerate(y) if v==0]
            pos_inds = [i for i, v in enumerate(y) if v==1]

            n_neg = 5000 - len(pos_inds)
            neg_inds = sample(neg_inds, n_neg)
            inds = sorted(neg_inds + pos_inds)
            X = X.iloc[inds,:]
            y = y[inds]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.66666, random_state=42)

        Xtrain_fname = join(XY_CACHE_FOLDER, "dfXtrain_%d_small.pickle" % uid)
        Xvalid_fname = join(XY_CACHE_FOLDER, "dfXvalid_%d_small.pickle" % uid)
        Xtest_fname = join(XY_CACHE_FOLDER, "dfXtestv_%d_small.pickle" % uid)
        ys_fname = join(XY_CACHE_FOLDER, "ysv_%d_small.pickle" % uid)

        X_train.to_pickle(Xtrain_fname)
        X_valid.to_pickle(Xvalid_fname)
        X_test.to_pickle(Xtest_fname)
        pickle.dump((y_train, y_valid, y_test), open(ys_fname, 'wb'))

        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def load_small_validation_dataframe(self, uid):
        Xtrain_fname = join(XY_CACHE_FOLDER, "dfXtrain_%d_small.pickle" % uid)
        Xvalid_fname = join(XY_CACHE_FOLDER, "dfXvalid_%d_small.pickle" % uid)
        Xtest_fname = join(XY_CACHE_FOLDER, "dfXtestv_%d_small.pickle" % uid)
        ys_fname = join(XY_CACHE_FOLDER, "ysv_%d_small.pickle" % uid)

        X_train = pd.read_pickle(Xtrain_fname)
        X_valid = pd.read_pickle(Xvalid_fname)
        X_test = pd.read_pickle(Xtest_fname)
        y_train, y_valid, y_test = pickle.load(open(ys_fname, 'rb'))

        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def load_or_create_dataframe(self, uid):
        Xtrain_fname = join(XY_CACHE_FOLDER, "dfXtrain_%d.pickle" % uid)
        Xtest_fname = join(XY_CACHE_FOLDER, "dfXtest_%d.pickle" % uid)
        ys_fname = join(XY_CACHE_FOLDER, "ys_%d.pickle" % uid)
        exists = False
        if os.path.exists(Xtrain_fname):
            try:
                X_train = pd.read_pickle(Xtrain_fname)
                X_test = pd.read_pickle(Xtest_fname)
                y_train, y_test = pickle.load(open(ys_fname, 'rb'))
                exists = True
            except Exception as e:
                pass

        if not exists:
            s = open_session()
            user = s.query(User).get(uid)
            neighbours = get_level2_neighbours(user, s)
            # remove central user from neighbours
            neighbours = [u for u in neighbours if u.id != user.id]

            # Fetch tweet universe (timelines of ownuser and neighbours)
            tweets = set(user.timeline)
            for u in neighbours:
                tweets.update(u.timeline)

            # exclude tweets from central user or not in Spanish
            tweets = [t for t in tweets if t.author_id != uid]  # and t.lang == 'es']

            tweet_ids = [t.id for t in tweets]
            neighbour_ids = [u.id for u in neighbours]
            X, y = self.extract_features(tweets, neighbours, user)
            s.close()

            X = pd.DataFrame(data=X, index=tweet_ids, columns=neighbour_ids)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            X_train.to_pickle(Xtrain_fname)
            X_test.to_pickle(Xtest_fname)
            pickle.dump((y_train, y_test), open(ys_fname, 'wb'))

        return X_train, X_test, y_train, y_test

Dataset = _Dataset()