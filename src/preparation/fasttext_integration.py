import os

# from preparation.utils import NLPFeatures
from preparation.nlp_features import NLPFeatures

FASTTEXT = './preparation/fasttext'
MODEL = "../model/third_party/fasttext/wiki.es.bin"
INPUT_FILE = "preparation/tweets.in"
OUTPUT_FILE = "preparation/tweets.out"


class FTEXT(object):
    def __init__(self):
        super(FTEXT, self).__init__()
        self.tweets = None

    def prepare(self):
        if not self.tweets:
            raise Exception("No tweets to evaluate")
        with open(INPUT_FILE, "w+") as f:
            for tw in self.tweets:
                text = NLPFeatures.preprocess_mati(tw.text, lemma=False)
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
        self.tweets = tweets
        self.prepare()
        self.run()
        return self.read()


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
