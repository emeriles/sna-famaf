import sqlite3


def _dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


TXT_BD_PATH = '../data/processed/txt_db.db'


class _DBHandler(object):

    def __init__(self, db_path=TXT_BD_PATH):
        """DBHandler initialization"""
        self.conn = None  # sqlite3.connect(BD_PATH)
        # self.conn.row_factory = None  # _dict_factory
        self.c = None  # self.conn.cursor()
        self.db_path = db_path

    def execute(self, query):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = _dict_factory
        self.c = self.conn.cursor()

        self.c.execute(query)
        result = self.c.fetchall()
        self.conn.commit()

        if len(result) == 0:
            return None

        return result

    def _query_one(self, query):
        result = self.execute(query)
        if not result:
            return None
        elif len(result) == 1:
            return result[0]
        else:
            raise Exception('More than one element returned on query!')

    def get_current_db_version(self):
        try:
            query = "SELECT * from version ORDER BY id DESC LIMIT 1"
            version = self._query_one(query)
        except sqlite3.OperationalError:
            print('No version was found. Created empty database at {}'.format(self.db_path))
            version = 0
        return version

    def get_features_by_id(self, tweets):
        if isinstance(tweets, str) or isinstance(tweets, int):
            tweets = [str(tweets)]
        feats = [self.execute("select features from tweets where id = '{}'".format(t)) for t in tweets]
        feats = [f[0]['features'] for f in feats]
        return feats

    @staticmethod
    def recreate_db():
        import os
        print('removing old db')
        os.remove(TXT_BD_PATH)
        DBHandler.execute(
            """
            CREATE TABLE IF NOT EXISTS tweets (
                    id text,
                    features text
                );
            """
        )
        print('created new db. {}'.format(TXT_BD_PATH))
        from processing.db_csv import _Dataset
        from preparation.fasttext_integration import OUTPUT_FILE

        tweet_ids = _Dataset.get_texts_id_str()
        print('loaded necessary data. now moving to db...')

        conn = sqlite3.connect(DBHandler.db_path)
        # conn.row_factory = _dict_factory
        c = conn.cursor()

        # result = self.c.fetchall()
        # self.conn.commit()

        to_process = full_n = len(tweet_ids)
        percentage = 0
        c.execute('PRAGMA synchronous = OFF')
        conn.commit()

        for tid, features in zip(tweet_ids, open(OUTPUT_FILE)):
            features = features.strip()
            query = """INSERT INTO tweets VALUES ('{id}', '{features}')""".format(id=tid, features=features)
            # DBHandler.execute(query)
            c.execute(query)

            to_process -= 1
            percentage = 100 - int(to_process / full_n * 100)
            print(
                'Avance: %{}'.format(percentage), end='\r'
            )
        conn.commit()
        print('done creating db.')


DBHandler = _DBHandler()
