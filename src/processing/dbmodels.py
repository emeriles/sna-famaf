from sqlalchemy import create_engine, Table, Column, ForeignKey
import pandas as pd
from sqlalchemy import (Integer, String, DateTime, Boolean)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy_utils.functions import drop_database, database_exists, create_database
import time

from processing.preprocess_csv import CSVDataframe
from settings import SQLITE_CONNECTION, CSV_CUTTED, NX_GRAPH_PATH

# DATE_LOWER_LIMIT = datetime(year=2015, month=8, day=24)
#
# DATE_UPPER_LIMIT = datetime(year=2015, month=9, day=24)


Base = declarative_base()


def db_connect(connection=SQLITE_CONNECTION):
    """
    Performs database connection using database settings from settings.py.
    Returns sqlalchemy engine instance
    """
    return create_engine(connection)

def open_session(connection=SQLITE_CONNECTION, engine=None):
    if engine is None:
        engine = db_connect(connection)
    Session = sessionmaker(bind=engine)
    session = Session()

    return session


def create_tables(engine):
    print('creating tables')
    DeclarativeBase.metadata.create_all(engine)

def initialize_db():
    engine = db_connect()
    if database_exists(engine.url):
        drop_database(engine.url)
    create_database(engine.url)
    create_tables(engine)
    

def create_tables(engine):
    Base.metadata.create_all(engine)

users_timeline = Table(
    "users_timeline",
    Base.metadata,
    Column("fk_user", Integer, ForeignKey("users.id")),
    Column("fk_tweet", Integer, ForeignKey("tweets.id")),
)

users_retweets = Table(
    "users_retweets",
    Base.metadata,
    Column("fk_user", Integer, ForeignKey("users.id")),
    Column("fk_retweet", Integer, ForeignKey("tweets.id")),
)

users_favs = Table(
    "users_favs",
    Base.metadata,
    Column("fk_user", Integer, ForeignKey("users.id")),
    Column("fk_fav", Integer, ForeignKey("tweets.id")),
)

users_follows = Table(
    "users_follows",
    Base.metadata,
    Column("fk_user_follows", Integer, ForeignKey("users.id")),
    Column("fk_user_followed", Integer, ForeignKey("users.id")),
)

class Tweet(Base):
    """SQLAlchemy Tweet model"""
    __tablename__ = "tweets"
    id = Column('id', Integer, primary_key=True)
    author_id = Column('author_id', Integer)
    created_at = Column('created_at', DateTime)
    retweet_count = Column('retweet_count', Integer)
    favorite_count = Column('favorite_count', Integer)
    text = Column('text', String(300))
    lang = Column('lang', String(2))
    is_quote_status = Column('is_quote_status', Boolean)

TWEET_FIELDS = [c.name for c in Tweet.__table__.columns if c.name != 'author_id']


class User(Base):
    """SQLAlchemy Hotel model"""
    __tablename__ = "users"
    id = Column('id', Integer, primary_key=True)
    username = Column('username', String)
    is_authorized = Column('is_authorized', Boolean, default=True)

    timeline = relationship(
        "Tweet",
        backref="users_posted",
        secondary=users_timeline
    )

    favs = relationship(
        "Tweet",
        backref="users_faved",
        secondary=users_favs
    )

    retweets = relationship(
        "Tweet",
        backref="users_retweeted",
        secondary=users_retweets
    )

    def fetch_timeline(self, session, df):
        print("Saving timeline for user %d" % self.id)
        # authenticating here ensures a different set of credentials
        # everytime we start processing a new county, to prevent hitting the rate limit
        self.timeline = []
        self.retweets = []

        # page = 1
        # done = False
        # while not done:
        #     TW_API = API_HANDLER.get_fresh_connection()
        #     try:
        #         tweets = TW_API.user_timeline(user_id=self.id, page=page)
        #     except Exception, e:
        #         if e.message == u'Not authorized.':
        #             self.is_authorized = False
        #             break
        #         else:
        #             print("Error: %s" % e.message)
        #             print "waiting..."
        #             time.sleep(10)
        #             continue
        # dtypes = {
        #     'user__id_str': str,
        #     'id_str': str,
        #     'text': str,
        #     'retweeted_status__id_str': str,
        #     'retweeted_status__user__id_str': str,
        #     'retweet_count': int,
        #     'quoted_status_id_str': str,
        # }
        # df = pd.read_csv(csv_path, dtype=dtypes)
        # ## FILTER ON USER
        id_s = str(self.id)
        # print(id_s)
        # print(type(id_s))
        # print(df['user.id_str'] == '51864127')
        df_filtered = df[(df['user__id_str'] == id_s) | (df['retweeted_status__user__id_str'] == id_s)]
        # for t in df:
        #     if not tweets:
        #         # All done
        #         break
        #     else:
        for idx, t in df_filtered.iterrows():
            # if t.created_at > DATE_UPPER_LIMIT:
            #     continue
            # elif t.created_at > DATE_LOWER_LIMIT:
            isretweet = False
            if not pd.isna(getattr(t, 'retweeted_status__id_str')):
                t_id = getattr(t, 'retweeted_status__id_str')
                isretweet = True
            else:
                t_id = t.id_str

            t_id = int(t_id)
            tweet = session.query(Tweet).get(t_id)
            pepe = getattr(t, 'retweeted_status__created_at') if hasattr(t, 'retweeted_status__created_at') else t.created_at
            if not tweet:
                tweet_data = {
                    'id': t_id,
                    'created_at': pd.to_datetime(pepe) if not pd.isna(pepe) else None,
                    'retweet_count': 0,
                    'favorite_count': 0,
                    'text': t.text,
                    'lang': '',
                    'is_quote_status': False,
                }
                # tweet = Tweet(**{f: t.__getattribute__(f) for f in TWEET_FIELDS})
                tweet = Tweet(**tweet_data)
                pepe2 = getattr(t, 'retweeted_status__user__id_str') if hasattr(t, 'retweeted_status__user__id_str') else getattr(t, 'user__id_str')
                tweet.author_id = pepe2
                session.add(tweet)
            if isretweet:
                self.retweets.append(tweet)
            self.timeline.append(tweet)
            # if t.favorited:
            #     self.favs.append(tweet)
            # else:
            #     done = True
            #     break
        # page += 1  # next page


        # elapsed_time =  time.time() - start_time
        # print "Done. Took %.1f secs to fetch %d tweets" % (elapsed_time, len(self.timeline))
        session.commit()
        
        return self.timeline


def reset_sqlite_db():
    initialize_db()
    csv_path = CSV_CUTTED
    
    import networkx as nx
    graph = nx.read_gpickle(NX_GRAPH_PATH)
    user_ids = graph.nodes()
    users = [User(id=int(uid), username=uid) for uid in user_ids]

    # TW = API_HANDLER.get_fresh_connection()
    # for i, u in enumerate(users):
    #     u.username = TW.get_user(u.id).name
    #     if (i + 1) % 20 == 0:
    #         TW = API_HANDLER.get_fresh_connection()

    session = open_session()
    session.add_all(users)
    session.close()

    to_process = len(users)
    csv_df = CSVDataframe(csv_path)
    percentage = 0
    for user in users:
        to_process -= 1
        percentage = 100 - int(to_process / len(users) * 100)
        print(
            'Avance: %{}'.format(percentage)
        )
        user.fetch_timeline(session, csv_df.df)
        # user.fetch_favorites(session)
