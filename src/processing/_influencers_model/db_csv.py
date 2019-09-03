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
from settings import CSV_CUTTED, JSON_TEXTS, XY_CACHE_FOLDER, NX_SUBGRAPH_PATH, ALPHA, INFLUENCE_POINTS, \
    MAX_INFLUENCERS_PERCENT


class _DatasetInfluencersModel(_Dataset):

    def __init__(self, csv_path=CSV_CUTTED, txt_path=JSON_TEXTS, delta_minutes_filter=None):
        super().__init__(csv_path=csv_path, txt_path=txt_path, delta_minutes_filter=delta_minutes_filter)
        self.relevant_users = None
        self.df_filtered = None
        # self.get_tweets_universe(0, [0])
        self.influence_points = None

        self.influencers = None
        self.target_users = None
        self.influencers_ids = None
        self.tweets = None
        self.tt_min_score = None
        self.lda = None
        self.tw_lda = None
        self.fasttext = None

        # self.get_tt_min_score()

    def load_nx_subgraph(self):
        return nx.read_gpickle(NX_SUBGRAPH_PATH)
    #
    # def get_level2_neighbours(self, user_id):
    #     """
    #     An ordered list of up to level-2 neighbours is created
    #     (followed and their followed)
    #     """
    #     g = self.load_nx_subgraph()
    #     uid = str(user_id)
    #
    #     neighbourhood = set(list(g.successors(uid)))
    #
    #     for nid in g.successors(uid):
    #         neighbourhood.update(g.successors(nid))
    #
    #     # Remove None elements and own user (TODO: see why this happens)
    #     neighbour_users = [u_id for u_id in neighbourhood if u_id and u_id != user_id]
    #
    #     return neighbour_users

    def get_relevant_users(self):
        """Return users with more than 10 retweets on the dataset."""
        if self.df.empty:
            self._load_df()
        only_rts = self.df[self.df.retweeted_status__id_str.notna()]
        rts_by_user = self.df.id_str.groupby(only_rts.user__id_str).count()
        users_considered = rts_by_user[rts_by_user > 10].index
        print('Users to be considered = {}'.format(len(users_considered)))
        self.relevant_users = users_considered
        return users_considered

    def load_tweets_filtered(self, min_rt_allowed=4, percentage=100):
        if self.df.empty:
            self._load_df()
        print("Loading tweets from db...")
        from processing._influencers_model.influence import InfluenceActions
        self.influence_points = InfluenceActions.load_influencers_from_pickle(INFLUENCE_POINTS)
        if not self.target_users:
            self.load_influencers_id_list(random=False)

        tweets = np.empty((0, 2))

        df_relevant_users = self.df[self.df.user__id_str.isin(self.target_users)]

        rts_counts = df_relevant_users.retweeted_status__id_str.groupby(df_relevant_users.retweeted_status__id_str).count()
        rts_ids_filtered = list(rts_counts[rts_counts > min_rt_allowed].index)

        df_filtered = df_relevant_users[df_relevant_users.retweeted_status__id_str.isin(rts_ids_filtered)]

        self.tweets = np.concatenate((tweets, df_filtered.loc[:, ('retweeted_status__id_str', 'created_at')]))

        # self.tweets = (self.s.query(Tweet)
        #                .join(Tweet.users_retweeted)
        #                .group_by(Tweet)
        #                .filter(User.id.in_(self.target_users))
        #                .filter(Tweet.lang=="es")
        #                .having(func.count(User.id) >= min_rt_allowed)
        #                #.filter(~Tweet.id.in_(self.ommited))
        #                .order_by(func.random())
        #                .all())

        if percentage != 100:
            percentage = percentage / 100.0
            limit = int(percentage * len(self.tweets))
            self.tweets = self.tweets[:limit]
        limit = int(.75 * len(self.tweets))
        self.train_tweets = self.tweets[:limit]
        self.test_tweets = self.tweets[limit:]
        print(len(self.train_tweets))
        print(len(self.test_tweets))
        print("\-Loaded {} tws [{} train, {} test]"
              .format(len(self.tweets),
                      len(self.train_tweets),
                      len(self.test_tweets))
              )

    def get_tt_min_score(self):
        if self.df.empty:
            self._load_df()
        rts_counts = self.df.retweeted_status__id_str.groupby(self.df.retweeted_status__id_str).count()
        rts_counts_filtered = rts_counts[rts_counts > 4].values

        self.tt_min_score = np.percentile(rts_counts_filtered, 90)

        print("\-TT MIN SCORE {}".format(self.tt_min_score))

    def load_influencers_id_list(self, number_of_influencers=1000, random=False):
        from processing._influencers_model.influence import InfluenceActions
        self.influence_points = InfluenceActions.load_influencers_from_pickle(INFLUENCE_POINTS)
        print("Getting best {} influencers of {}".format(number_of_influencers,
                                                         len(self.influence_points))
              )
        filtered_i = list(filter(lambda x: x.user is not None, self.influence_points))
        sorted_i = sorted(filtered_i, key=lambda x: ALPHA * x.activity + (1-ALPHA) * x.centrality,
                          reverse=True)
        influencers_ids = [i.user_id for i in sorted_i]
        self.influencers_ids = influencers_ids
        if random:
            print("Every day im shuffeling")
            import random
            random.shuffle(self.influencers_ids)
        self.influencers = self.influencers_ids
        self.target_users = self.influencers_ids[int(len(influencers_ids) * MAX_INFLUENCERS_PERCENT):]
        if number_of_influencers:
            self.influencers_ids = influencers_ids[:number_of_influencers]
            #self.target_users = influencers_ids[number_of_influencers:]
        return influencers_ids

    def get_influencers_id_list(self, number_of_influencers):
        if number_of_influencers:
            self.influencers_ids = self.influencers[:number_of_influencers]
            #self.target_users = influencers_ids[number_of_influencers:]
        return self.influencers_ids

    # def _extract_features_full(self, tweets, neighbour_users, own_user):
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
    #     X = scipy.sparse.lil_matrix((nrows, nfeats))
    #     #         X = np.empty((nrows, nfeats), dtype=np.int8)
    #     #         print('X SIZE (MB): ', X.nbytes  * 1000000)
    #     y = np.empty(nrows)
    #     print('Extracting features Optimized. X shape is :', X.shape)
    #
    #     to_process = tweets.shape[0]
    #     for j, u in enumerate(neighbour_users):
    #         to_process -= 1
    #         percentage = 100.0 - ((to_process / tweets.shape[0]) * 100)
    #         print('Avance: %{}'.format(percentage), end='\r')
    #
    #         n_tl_filtered = self.get_user_timeline(u)
    #         col = np.isin(tweets[:, 0], n_tl_filtered[:, 0])
    #         #             print(X[:, j].shape, col.reshape((11,1)))
    #         X[:, j] = col.reshape((nrows, 1))
    #         # print(X[:, j])
    #
    #     own_tl_filtered = self.get_user_timeline(own_user)
    #     y = np.isin(tweets[:, 0], own_tl_filtered[:, 0])
    #     end = datetime.datetime.now() - start
    #     print('Done Extracting Features', end)
    #     return X, y

    def get_user_timeline(self, uid):
        """
        Returns [(tweet_id, creted_at)] for a given user id or list of users ids
        :param uid:
        :return:
        """
        if self.df.empty:
            self._load_df()
        if isinstance(uid, str) or isinstance(uid, int):
            uid = [str(uid)]
        filtered = self.df[(self.df.user__id_str.isin(uid))]
        tweets = filtered.copy()
        own_tweets = tweets.loc[:, ('id_str', 'created_at')]
        rts = tweets.loc[:, ('retweeted_status__id_str', 'retweeted_status__created_at')]
        rts.rename({'retweeted_status__id_str': 'id_str',
                    'retweeted_status__created_at': 'created_at'},  ########## ??????????????
                   axis='columns', inplace=True)
        timeline = pd.concat([own_tweets, rts]).dropna().drop_duplicates(subset='id_str').values
        return timeline

    def extract_features(self, dataset="train", with_influencers=True):
        if self.tweets is None:
            print("--No tweets found")
            return
        if dataset == "train":
            dataset = self.train_tweets
            print("Extracting features on train data")
        else:
            dataset = self.test_tweets
            print("Extracting features on test data")
        nrows = len(dataset)
        nfeats = 0
        if with_influencers:
            nfeats = len(self.influencers_ids)
        if self.lda is not None:
            X = np.zeros((nrows, nfeats + self.lda.model.num_topics))
        elif self.tw_lda is not None:
            nfeatsplus = nfeats + int(self.tw_lda.settings['topics'])
            X = np.zeros((nrows, nfeatsplus))
        elif self.fasttext is not None:
            ft_sentence_vectors = self.fasttext.get_embeddings(dataset)
            X = np.zeros((nrows, nfeats + 300))
        else:
            X = np.zeros((nrows, nfeats))

        to_process = X.shape[0]
        print('Extracting features X shape is :', X.shape)
        for j, u in enumerate(self.influencers_ids):
            to_process -= 1
            percentage = 100.0 - ((to_process / X.shape[0]) * 100)
            print('Avance: %{}'.format(percentage), end='\r')

            n_tl_filtered = self.get_user_timeline(u)
            col = np.isin(dataset[:, 0], n_tl_filtered[:, 0])
            #             print(X[:, j].shape, col.reshape((11,1)))
            # print('SHAPE X[:, j]: {}...'.format(X[:, j].shape))
            # print('SHAPE col.reshape((nrows, 1)): {}...'.format(col.reshape((nrows, 1)).shape))
            # print('SHAPE col: {}...'.format(col.shape))
            X[:, j] = col
            # print(X[:, j])

        rts_counts = self.df.retweeted_status__id_str.groupby(self.df.retweeted_status__id_str).count()

        if self.tt_min_score is None:
            self.get_tt_min_score()
        y_ = rts_counts[dataset[:, 0]]
        y = y_ > self.tt_min_score
        y.replace(to_replace=[True, False], value=[1, 0], inplace=True)

        # y[idx] = 1 if len(users_with_tweet) >= self.tt_min_score else 0

        # for idx, tweet in enumerate(dataset):
        #     users_with_tweet = tweet.users_retweeted
        #     users_ids = [str(u.id) for u in users_with_tweet]
        #     if with_influencers:
        #         for idy, influencer_id in enumerate(self.influencers_ids):
        #             X[idx, idy] = 1 if str(influencer_id) in users_ids else 0
        #     if self.lda:
        #         for idt, topic in enumerate(self.lda.get_topic_features(tweet.text)):
        #             X[idx, nfeats+idt] = 1 if topic > 0.25 else 0
        #     elif self.tw_lda:
        #         topic = self.tw_lda.topics[str(tweet.id)]
        #         X[idx, nfeats+topic] = 1
        #     elif self.fasttext:
        #         X[idx, nfeats:] = ft_sentence_vectors[idx].split(" ")[:-1]
        #     y[idx] = 1 if len(users_with_tweet) >= self.tt_min_score else 0

        # if sum(sum(X)) == 0:
        #     raise Exception("Zero matrix X")
        # if sum(y) == 0:
        #     raise Exception("Zero matrix y")
        return X, y

    # def extract_features(self, tweets, neighbour_users, own_user, timedelta=None):
    #     """
    #     Given tweets and neighbour_users, we extract
    #     'neighbour activity' features for each tweet
    #
    #     These are obtained as follows:
    #         - for each of these users a boolean feature is created
    #         indicating if the tweet is authored/retweeted by that user and is comprehended in timedelta window
    #     """
    #     if not timedelta:
    #         return self._extract_features_full(tweets, neighbour_users, own_user)

    # def load_or_create_dataset(self, uid, delta_minutes_filter):
    #     self.delta_minutes = delta_minutes_filter
    #     self.df = pd.DataFrame()
    #     model_name = self.__class__.__name__
    #     fname = join(XY_CACHE_FOLDER, "dataset_{}_{}_{}.pickle".format(model_name, uid, self.delta_minutes))
    #     if os.path.exists(fname):
    #         dataset = pickle.load(open(fname, 'rb'))
    #     else:
    #         self._load_df()
    #         uid = str(uid)
    #         neighbours = self.get_neighbourhood(uid)
    #         # remove selected user from neighbours
    #         neighbours = [u for u in neighbours if u != uid]
    #
    #         tweets = self.get_tweets_universe(uid, neighbours)
    #
    #         X, y = self.extract_features(tweets, neighbours, uid)
    #
    #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    #         X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.66666, random_state=42,
    #                                                             stratify=y_test)
    #         dataset = (X_train, X_test, X_valid, y_train, y_test, y_valid)
    #
    #         pickle.dump(dataset, open(fname, 'wb'))
    #
    #     (X_train, X_test, X_valid, y_train, y_test, y_valid) = dataset
    #     return dataset


DatasetInfluencersModel = _DatasetInfluencersModel()
