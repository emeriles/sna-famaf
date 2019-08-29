import datetime

import pandas as pd
import numpy as np

from settings import CSV_CUTTED, JSON_TEXTS


class _Dataset(object):
    """Base class for different datasets"""

    def __init__(self, csv_path=CSV_CUTTED, txt_path=JSON_TEXTS, delta_minutes_filter=None):
        self.csv_path = csv_path
        self.txt_path = txt_path
        self.df: pd.DataFrame = pd.DataFrame()
        self.text_df: pd.DataFrame = pd.DataFrame()
        self.delta_minutes = delta_minutes_filter

    def get_most_active_users(self, N=1000, just_ids=True):
        if self.df.empty:
            self._load_df()
        most_active = sorted(self.df.user__id_str.groupby(self.df.user__id_str).count().iteritems(),
                             reverse=True, key=lambda x: x[1])
        print('Sample of first 20 most active users:')
        print(list(most_active)[:20])
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
                (df.created_at - df.retweeted_status__created_at <= datetime.timedelta(minutes=self.delta_minutes)) |
                np.isnat(df.retweeted_status__created_at)]
            df = df_filtered.copy()

        self.df = df
        print('Done loading df. DF shape is :{} (Original: {}) \t\tTime delta is: {} mins'. \
              format(df.shape, original_shape, self.delta_minutes))

        # self._load_text_df()
        return df

    def _load_text_df(self):
        print('Loading text df: {}'.format(self.txt_path))
        df = pd.read_json(self.txt_path)
        df.drop_duplicates(subset='id_str', inplace=True)
        df.drop(['_id'], axis=1, inplace=True)
        df.set_index('id_str', inplace=True)
        self.text_df = df
        return df

    def get_all_users(self):
        if self.df.empty:
            self._load_df()
        return self.df.user__id_str.unique()

    def get_user_timeline(self, uid, with_original=True, with_retweets=True):
        """
        Returns [(tweet_id, creted_at)] for a given user id or list of users ids
        :param with_original:
        :param with_retweets:
        :param uid:
        :return:
        """
        if self.df.empty:
            self._load_df()
        if isinstance(uid, str) or isinstance(uid, int):
            uid = [str(uid)]
        filtered = self.df[(self.df.user__id_str.isin(uid))]
        tweets = filtered.copy()

        if with_original:
            own_tweets = tweets[pd.isna(tweets.retweeted_status__id_str)]
            own_tweets = own_tweets.loc[:, ('id_str', 'created_at')]
        else:
            own_tweets = np.empty((0, 2))

        if with_retweets:
            rts = tweets[pd.notna(tweets.retweeted_status__id_str)]
            rts = rts.loc[:, ('retweeted_status__id_str', 'created_at')]
            rts.rename({'retweeted_status__id_str': 'id_str',
                        # 'retweeted_status__created_at': 'created_at'
                        },
                       axis='columns', inplace=True)
        else:
            rts = np.empty((0, 2))

        timeline = pd.concat([own_tweets, rts]).dropna().drop_duplicates(subset='id_str').values
        return timeline

    def get_tweets_universe(self, uid, neighbours):
        """Override by child classes"""
        raise NotImplementedError()

    def get_neighbourhood(self, uid):
        """override with child classes.
        For a given user, returns all users to be used as features in retweeting action"""
        raise NotImplementedError()

    def load_or_create_dataset(self, *args, **kwargs):
        """Override with child classes.
        Builds complete X matrix. returns a dataset in a tuple form:
        dataset = (X_train, X_test, X_valid, y_train, y_test, y_valid)
        """
        raise NotImplementedError()

    def get_tweets_texts(self):
        return self.text_df
