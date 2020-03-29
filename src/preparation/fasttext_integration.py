import os

# from preparation.utils import NLPFeatures
import pickle

from preparation.nlp_features import NLPFeatures
import pandas as pd
from processing.db_csv import _Dataset
from settings import TXT_EMBEDDINGS_SERIES

FASTTEXT = './preparation/fasttext'
MODEL = "../model/third_party/fasttext/wiki.es.bin"
INPUT_FILE = "preparation/tweets.in"
OUTPUT_FILE = "preparation/tweets.out"


class FTEXT(object):
    def __init__(self):
        super(FTEXT, self).__init__()
        self.tweets = None

    def prepare(self):
        # if not self.tweets:
        #     raise Exception("No tweets to evaluate")
        with open(INPUT_FILE, "w+") as f:
            for tw in self.tweets:
                text = NLPFeatures.preprocess_mati(tw, lemma=False)
                f.write(text + "\n")
        with open(INPUT_FILE, "r+") as f:
            assert (len(self.tweets) == len(f.readlines()))

    def run(self):
        if not self.tweets:
            raise Exception("No tweets to evaluate")
        print('Running external fasttext')
        cmd = ("{} print-sentence-vectors {} < {} > {}"
               .format(FASTTEXT, MODEL, INPUT_FILE, OUTPUT_FILE))
        print(cmd)
        os.system(cmd)

    def read(self):
        with open(OUTPUT_FILE, "r") as f:
            lines = f.readlines()
        return lines

    def get_embeddings(self, tweets):
        """
        tweets is a list of strings.
        :param tweets:
        :return:
        """
        print('Getting embeddings for {} tweets.'.format(len(tweets)))
        self.tweets = tweets
        self.prepare()
        self.run()
        embeddings = self.read()
        print('Done gettin embeddings. There are {}.'.format(len(embeddings)))
        print('example: ', embeddings[:20])
        return embeddings


class FTextActions(object):
    @staticmethod
    def build_ftext_features():
        # load tweets
        tweets_w_ids = FTextActions.get_tweets_id_text()
        print('There are {} tweets_w_ids'.format(len(tweets_w_ids)))
        tweets_txt = tweets_w_ids[:, 1]
        ftext = FTEXT()
        txt_embeddings = ftext.get_embeddings(tweets_txt)

        # save txt embeddings in pandas.Series with index on id_str
        tweets_id_str = tweets_w_ids[:, 0]
        embeddings_series = pd.Series(data=txt_embeddings, index=tweets_id_str)
        with open(TXT_EMBEDDINGS_SERIES, 'wb') as f:
            pickle.dump(embeddings_series, f)

    @staticmethod
    def load_embeddings_series():
        with open(TXT_EMBEDDINGS_SERIES, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def get_tweets_id_text():
        """
        return list of id_ and its text. It must consider if it is a retweet, it should have the id of the retweet.
        :return:
        """
        return _Dataset.get_texts_id_str()

"""
    timeline counts info

import math
acu = 0
mx = 0
mn = math.inf
values = [-1] * s.query(User).count()
for i, u in enumerate(s.query(User)):
    acu += len(u.timeline)
    if mx < len(u.timeline):
        mx = len(u.timeline)
        mx_u = u
    # mx = max(mx, g.out_degree(u))
    mn = min(mn, len(u.timeline))
acu * 1.0 / s.query(User).count(), mx, mn, mx_u
# Out[13]: (353.7542471042471, 3132, 0, <dbmodels.User at 0x7f0224960940>)
"""
