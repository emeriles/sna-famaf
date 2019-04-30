import numpy as np
import pickle
import os
import pandas as pd
from os.path import join
from os import remove
from random import sample
from sklearn.model_selection import train_test_split

from processing.dbmodels import open_session, User
# from processing.utils import get_level2_neighbours
from settings import XY_CACHE_FOLDER


class Datasets(object):

    def __init__(self, df_path):
        self.df_path = df_path

    def get_active_user(self, session):
        rtcounts = {u: len(u.retweets) for u in s.query(User).all()}
        most_active = sorted(rtcounts.items(), key=lambda x: -x[1])

        return most_active

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


    def extract_features(self, tweets, neighbour_users, own_user):
        '''
            Given tweets and neighbour_users, we extract
            'neighbour activity' features for each tweet

            These are obtained as follows:
                - for each of these users a boolean feature is created
                indicating if the tweet is authored/retweeted by that user
        '''
        nrows = len(tweets)
        nfeats = len(neighbour_users)
        X = np.empty((nrows, nfeats))
        y = np.empty(nrows)

        own_tl_ids = [t.id for t in own_user.timeline]
        for j, u in enumerate(neighbour_users):
            tl_ids = [t.id for t in u.timeline]
            for i, t in enumerate(tweets):
                X[i, j] = 1 if t.id in tl_ids else 0  # <<<<<<<< aquí la comparación de ids iguales

        for i, t in enumerate(tweets):
            y[i] = 1 if t.id in own_tl_ids else 0

        return X, y

    def get_neighbourhood(self, uid):
        s = open_session()
        user = s.query(User).get(uid)
        neighbours = get_level2_neighbours(user, s)
        # remove central user from neighbours
        neighbours = [u for u in neighbours if u.id != user.id]

        return neighbours

    def load_or_create_dataset(self, uid):
        fname = join(XY_CACHE_FOLDER, "dataset_%d.pickle" % uid)
        if os.path.exists(fname):
            dataset = pickle.load(open(fname, 'rb'))
        else:
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
            tweets = [t for t in tweets if t.author_id != uid ]  # and t.lang == 'es'   all tweets in spanish?

            X, y = self.extract_features(tweets, neighbours, user)
            s.close()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            dataset = (X_train, X_test, y_train, y_test)

            pickle.dump(dataset, open(fname, 'wb'))


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
