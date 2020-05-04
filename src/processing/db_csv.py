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
        self.ids_texts = None
        self.as_seconds = False

    def get_most_active_users(self, N=1000, just_ids=True):
        if self.df.empty:
            self._load_df()
        most_active = sorted(self.df.retweeted_status__user__id_str.groupby(self.df.user__id_str).count().iteritems(),
                             reverse=True, key=lambda x: x[1])
        print('Sample of first 20 most active users:')
        print(list(most_active)[:20])
        if just_ids:
            return [id_ for id_, counts in most_active][:N]
        return most_active[:N]

    def _load_ftext_features(self):
        print('\tLoading fasttext features... (this should be done just once or so...)')
        # self.ftext_features_series = FTextActions.load_embeddings_series()  <- NO. 40 gb en memoria.
        # do al preprocessing ... like split(' ') . .. return np.array directly? no. ir results in 40gb structures.
        # just load file as strings, and dict_like structure to convert tweet_id_str -> row_number in string file
        from preparation.fasttext_integration import FTextActions
        # self.ftext_matrix = FTextActions.get_embeddings()
        # self.ftext_id_to_row =FTextActions.get_tweet_id_to_row_for_fasttext()
        self.ids_texts = FTextActions.get_tweets_id_text()

    def _get_embeddings_for_tweet(self, tweet_ids):
        pass
        if self.ids_texts is None:
            self._load_ftext_features()
        texts = self.ids_texts[tweet_ids]
        # if isinstance(tweet_ids, str) or isinstance(tweet_ids, int):
        #     tweet_ids = np.array([str(tweet_ids)])
        # rows = self.ftext_id_to_row[tweet_ids]
        from preparation.fasttext_integration import FTEXT
        ftext = FTEXT()
        embeddings = ftext.get_embeddings(tweets=texts)
        return embeddings
        # return self.ftext_matrix[rows, ]
        # Sí, pero ojo con self.ftext_matrix!


    def _load_df(self, central_uid=None):
        print('Loading df')
        dtypes = {
            'user__id_str': str,
            'id_str': str,
            'retweeted_status__id_str': str,
            'retweeted_status__user__id_str': str,
            'retweet_count': str,
            'quoted_status_id_str': str,
            'lang': str,
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
        #     df_filtered = df[np.isnat(df.retweeted_status__created_at) |
        #         (df.created_at - df.retweeted_status__created_at <= datetime.timedelta(minutes=self.delta_minutes))]
        #     df = df_filtered.copy()
        #
        self.df = df
        # print('Done loading df. DF shape is :{} (Original: {}) \t\tTime delta is: {} mins'. \
        #       format(df.shape, original_shape, self.delta_minutes))
        #
        # # self._load_text_df()
        return df

    @staticmethod
    def _load_text_df(filename_txt_json_path=JSON_TEXTS):
        print('Loading text df: {}'.format(filename_txt_json_path))

        df = pd.read_json(filename_txt_json_path, lines=True, dtype=False)
        print('Loaded raw text dataframe. Shape is: ', df.shape)

        # set correct id for our project
        df['correct_id'] = np.where(pd.isna(df.rt_st__id_str), df.id_str, df.rt_st__id_str)
        df.drop_duplicates(subset='correct_id', inplace=True)
        df['id_str'] = df['correct_id']

        return df

    def get_all_users(self):
        if self.df.empty:
            self._load_df()
        return self.df.user__id_str.unique()

    def get_user_timeline(self, uid, with_original=True, with_retweets=True, filter_timedelta=False):
        """
        Returns [(tweet_id, creted_at, retweeted_status__created_at)] for a given user id or list of users ids
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
            own_tweets = own_tweets.loc[:, ('id_str', 'created_at', 'retweeted_status__created_at')]
        else:
            own_tweets = pd.DataFrame()

        if with_retweets:
            if filter_timedelta and self.delta_minutes is not None:
                time_delta = datetime.timedelta(seconds=self.delta_minutes) if self.as_seconds else \
                    datetime.timedelta(minutes=self.delta_minutes)
                time_constraint = (tweets.created_at - tweets.retweeted_status__created_at) < time_delta
            else:
                time_constraint = True

            rts = tweets[pd.notna(tweets.retweeted_status__id_str) & time_constraint]
            rts = rts.loc[:, ('retweeted_status__id_str', 'created_at', 'retweeted_status__created_at')]
            rts.rename({'retweeted_status__id_str': 'id_str',
                        # 'retweeted_status__created_at': 'created_at'
                        },
                       axis='columns', inplace=True)
        else:
            rts = pd.DataFrame()

        timeline = pd.concat([own_tweets, rts]).dropna(subset=['id_str']).drop_duplicates(subset='id_str').values
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

    @staticmethod
    def get_texts_id_str():
        text_df = _Dataset._load_text_df()
        full = text_df[['id_str', 'text']].values
        just_ids = np.array(full[:, 0])
        return just_ids
        # return np.array((text_df[['id_str', 'text']].values)[:, 0])
