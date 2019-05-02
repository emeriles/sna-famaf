import numpy as np
import pandas as pd

from settings import CSV_RAW, CSV_CUTTED


class PreprocessCSV(object):

    def __init__(self, complete_csv_path=CSV_RAW):
        self.df_path: str = complete_csv_path
        self.df: pd.DataFrame = self._load_df()
        self.cut1: pd.DataFrame = None

    def _load_df(self):
        """
        Loads and FILTERS dataframe on time constraints with retweet-time elapsed from original tweet.
        :return:
        """
        print('Loading df')
        dtypes = {
            'user.id_str': str,
            'id_str': str,
            'retweeted_status.id_str': str,
            'retweeted_status.user.id_str': str,
            'retweet_count': int,
            'quoted_status_id_str': str,
        }
        df = pd.read_csv(self.df_path, dtype=dtypes)

        # parse dates
        datetime_cols = [c for c in df.columns if 'created_at' in c]
        for c in datetime_cols:
            df[c] = pd.to_datetime(df[c])

        # reemplazar nombre de columnas: . por __ para sintactic sugar de pandas.
        df.rename(columns=lambda x: x.replace('.', '__'), inplace=True)
        df.drop_duplicates(subset='id_str', inplace=True)
        print('Loading df finished')
        return df

    def df_cut1(self):
        """
        Returns a new dataframe with altered data: Recicling of tweets and included interaction among our users only.
        :param data_f:
        :return:
        """
        print('Performing df cut over {}'.format(self.df_path))
        # el filtro 1 va a ser un gran OR de:
        # OR item 1: tweets originales en este dataset (que no son RT de nada)
        data_f = self.df.copy(deep=True)
        original_tweets_on_ds = data_f.retweeted_status__id_str.isna()

        # OR item 2: tweets que son RT de un tweet de nuestros usuarios (un tweet de nuestros usuarios,
        # es un tweet original en este dataset)
        retweeted_by_my_users = data_f.retweeted_status__id_str.isin(self.df.id_str)

        # OR item 3: tweets que fueron retweeteados mas de 1 vez en este dataset.
        # para mantener consistencia en esta alteración el campo retweeted_status__created_at debe ser modificado,
        # usando el timestamp más temprano de los retweets de un mismo tweet.

        # obtener los conteos de tweets en el dataset
        rt_counts_on_ds = data_f.groupby(data_f.retweeted_status__id_str).count()
        # filtrar los tweets que tiene más de un retweet en el ds
        tweet_ids_w_more_than_one_rt = rt_counts_on_ds[rt_counts_on_ds.retweeted_status__user__id_str > 1]
        # obtener los ids de dichos tweets
        tweet_ids_w_more_than_one_rt = list(tweet_ids_w_more_than_one_rt.index)
        # seleccionar solo los que estan en el filtro
        tweet_w_more_than_one_rt = data_f.retweeted_status__id_str.isin(tweet_ids_w_more_than_one_rt)

        # Ahora vamos a reasignar los valores correctos a retweeted_status__created_at de los tweet_w_more_than_one_rt
        # para ello se crea un nuevo campo 'retweeted_status__new_created_at'

        # selecciono solo los tweets que voy a modificar
        aux_df = data_f[tweet_w_more_than_one_rt]
        # definimos una serie de pandas que sirve como diccionario: retweeted_status__id_str -> minimo timestamp
        # en el ds, SOLO para aquellos retweets 'reciclados' en este procesamiento.
        min_timestamps = aux_df.loc[aux_df.groupby(aux_df.retweeted_status__id_str).created_at.idxmin()]
        # para setear en data_f el nuevo campo adecuadamente, los NaT deben ser tratados. Se los pasa temporalmente
        # por un -1 -> nan.

        min_timestamps.set_index('retweeted_status__id_str', inplace=True)
        min_timestamps.loc['-1'] = np.nan

        # copiamos la columna retweeted_status__id_str, para reemplazarla por los new_created_at, llenando los NA con -1
        new_created_at_column = data_f.retweeted_status__id_str.fillna('-1').values
        # finalmente, pasamos la columna a los timestamps que calculamos, de acuerdo a su retweeted_status__id_str
        data_f['retweeted_status__new_created_at'] = min_timestamps.loc[new_created_at_column].created_at.values
        # reasignamos valores correctos para los campos retweeted_status__user_id
        data_f['retweeted_status__new_user__id_str'] = min_timestamps.loc[new_created_at_column].user__id_str.values
        # reasignamos valores correctos para los campos retweeted_status__id_str
        data_f['retweeted_status__new_id_str'] = min_timestamps.loc[new_created_at_column].index.values

        # ya tenemos a data_f con los valores "reciclados", pero faltan los originales (rt_status__created_at,
        # rt_status__id_str y rt_status__user_id_str que no fueron reciclados)
        # si la columna nueva de created_at es NaT, rellenamos con el valor de retweeted_status__created_at
        data_f['retweeted_status__new_created_at'] = \
            data_f.retweeted_status__new_created_at.fillna(data_f.retweeted_status__created_at)
        # si la columna nueva de new_id_str es '-1', rellenamos con nan,
        # para luego rellenar con el valor de rt_status__id_str, si se encontro nan
        data_f.retweeted_status__new_id_str = data_f.retweeted_status__new_id_str.replace('-1', value=np.nan)
        data_f['retweeted_status__new_id_str'] = \
            data_f.retweeted_status__new_id_str.fillna(data_f.retweeted_status__id_str)
        # si la columna nueva de new_user__id_str es NaT, rellenamos con el valor de retweeted_status__user__id_str
        data_f['retweeted_status__new_user__id_str'] = \
            data_f.retweeted_status__new_user__id_str.fillna(data_f.retweeted_status__user__id_str)

        # reasignamos los valores con los que vamos a trabajar
        data_f['retweeted_status__created_at'], data_f['retweeted_status__old_created_at'] = \
            data_f['retweeted_status__new_created_at'], data_f['retweeted_status__created_at']
        data_f['retweeted_status__user__id_str'], data_f['retweeted_status__old_user__id_str'] = \
            data_f['retweeted_status__new_user__id_str'], data_f['retweeted_status__user__id_str']
        data_f['retweeted_status__id_str'], data_f['retweeted_status__old_id_str'] = \
            data_f['retweeted_status__new_id_str'], data_f['retweeted_status__id_str']

        result = data_f[original_tweets_on_ds | retweeted_by_my_users | tweet_w_more_than_one_rt]
        use_cols = [
            'created_at', 'user__id_str', 'id_str', 'retweeted_status__id_str',
            'retweeted_status__user__id_str', 'retweeted_status__created_at', 'retweet_count', 'quoted_status_id_str'
        ]
        new_df = result[use_cols].copy(deep=True)
        self.cut1 = new_df
        print('Performing cut1 done.')
        return new_df

    def save_cut1(self, filename=CSV_CUTTED):
        print('Saving dataframe {}'.format(filename))
        self.cut1.to_csv(filename, index=False)

    @staticmethod
    def create_and_save_csv_cutted():
        p = PreprocessCSV()
        p.df_cut1()
        p.save_cut1()


class CSVDataframe(object):

    def __init__(self, df_path):
        self.df_path = df_path
        self.df = self._load_df()

    def _load_df(self):
        print('Loading CSV DF: ', self.df_path)
        dtypes = {
            'created_at': str,
            'user__id_str': str,
            'id_str': str,
            'retweeted_status__id_str': str,
            'retweeted_status__user__id_str': str,
            'retweeted_status__created_at': str,
            'retweet_count': str,
            'quoted_status_id_str': str,
            # 'retweeted_status__new_created_at': str,
            # 'retweeted_status__new_user__id_str': str,
            # 'retweeted_status__new_id_str': str,
            # 'retweeted_status__old_created_at': str,
            # 'retweeted_status__old_user__id_str': str,
            # 'retweeted_status__old_id_str': str,
        }
        df = pd.read_csv(self.df_path, index_col=False, dtype=dtypes)

        # parse dates
        # datetime_cols = [c for c in df.columns if 'created_at' in c]
        # for c in datetime_cols:
        #     df[c] = pd.to_datetime(df[c])

        # reemplazar nombre de columnas: . por __ para sintactic sugar de pandas.
        df.rename(columns=lambda x: x.replace('.', '__'), inplace=True)
        print('Loading df finished')
        return df
