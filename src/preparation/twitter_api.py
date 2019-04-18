from tweepy import AppAuthHandler, API
from tweepy.error import TweepError
from random import choice
from private_settings import AUTH_DATA
import time

# Used to switch between tokens to avoid exceeding rate limits
class APIHandler(object):
    """docstring for APIHandler"""
    def __init__(self, auth_data, max_nreqs=10):
        self.auth_data = auth_data
        self.index = choice(range(len(auth_data)))
        self.max_nreqs = max_nreqs

    def conn(self):
        if self.nreqs == self.max_nreqs:
            self.get_fresh_connection()
        else:
            self.nreqs += 1
        return self.conn_

    def get_fresh_connection(self):
        success = False
        while not success:
            try:
                self.index = (self.index + 1) % len(self.auth_data)
                d = self.auth_data[self.index]
                print("Switching to API Credentials #%d" % self.index)

                auth = AppAuthHandler(d['consumer_key'], d['consumer_secret'])
                self.conn_ = API(auth_handler=auth, wait_on_rate_limit=False, wait_on_rate_limit_notify=True)
                self.nreqs = 0
                return self.conn_
            except TweepError as e:
                print("Error trying to connect: %s" % e.message)
                time.sleep(10)

    def traer_seguidores(self, **kwargs):
        conns_tried = 0
        fids = []
        cursor = -1
        while cursor:
            try:
                fs, (_, cursor) = self.conn_.followers_ids(count=5000, cursor=cursor, **kwargs)
                fids += [str(x) for x in fs]
                # if not fids:
                #     # terminamos

                print("fetched %d followers so far." % len(fids))
            except TweepError as e:

                if not 'rate limit' in e.reason.lower():
                    raise e
                else:
                    conns_tried += 1
                    if conns_tried == len(self.auth_data):
                        nmins = 15
                        print(e)
                        print("Rate limit reached for all connections. Waiting %d mins..." % nmins)
                        time.sleep(60 * nmins)
                        conns_tried = 0 # restart count
                    else:
                        self.get_fresh_connection()

        return fids


    def traer_seguidos(self, **kwargs):
        conns_tried = 0
        fids = []
        cursor = -1
        while cursor:
            try:
                fs, (_, cursor) = self.conn_.friends_ids(count=5000, cursor=cursor, **kwargs)
                fids += [str(x) for x in fs]
                # print "fetched %d followers so far." % len(fids)
            except TweepError as e:
                if not 'rate limit' in e.reason.lower():
                    raise e
                else:
                    conns_tried += 1
                    if conns_tried == len(self.auth_data):
                        nmins = 15
                        print("Rate limit reached for all connections. Waiting %d mins..." % nmins)
                        time.sleep(60 * nmins)
                        conns_tried = 0 # restart count
                    else:
                        self.get_fresh_connection()
        return fids

    def traer_timeline(self, user_id, desde=None, hasta=None, dia=None, limite=None, since_id=None, screen_name=None):
        tweets = []
        page = 1
        if dia:
            desde = dia
            hasta = dia
        stop_sign = False

        while True and not stop_sign:
            print('tweets len: %d' % len(tweets), end="\r")
            if limite and len(tweets) >= limite:
                break

            try:
                page_tweets = self.conn_.user_timeline(user_id=user_id, page=page, since_id=since_id, screen_name=screen_name)
                if len(page_tweets) == 0:
                    break

                for tw in page_tweets:
                    if desde and tw.created_at.date() < desde:
                        stop_sign = True
                        break
                    if hasta and tw.created_at.date() > hasta:
                        continue

                    tweets.append(tw._json) # =dia or >= desde
                page += 1
            except TweepError as e:
                if 'Not authorized.' in e.reason:
                    print('Not authorized!')
                    break
                elif 'Sorry, that page does not exist.' in e.reason:
                    print('Not found!')
                    break
            except Exception as e:
                print("Error. Something really bad happened: %s" % e)
                print("waiting...")
                time.sleep(30)
                continue

        return tweets


    def statuses_lookup(self, twids, tweets=[]):
        for start in range(0, len(twids), 100):
            batch = twids[start: start + 100]
            try:
                tweets += self.conn_.statuses_lookup(batch)
                print("fetched %d tweets so far." % len(tweets))
            except TweepError as e:
                if not 'rate limit' in e.reason.lower():
                    raise e
                else:
                    conns_tried += 1
                    if conns_tried == len(self.auth_data):
                        nmins = 15
                        print(e)
                        print("Rate limit reached for all connections. Waiting %d mins..." % nmins)
                        time.sleep(60 * nmins)
                        conns_tried = 0 # restart count
                    else:
                        self.get_fresh_connection()

        return tweets

    def get_status(self, twid, **kwargs):
        conns_tried = 0
        while True:
            try:
                tweet = self.conn_.get_status(twid, **kwargs)
                return tweet
            except TweepError as e:
                if not 'rate limit' in e.reason.lower():
                    raise e
                else:
                    conns_tried += 1
                    if conns_tried == len(self.auth_data):
                        nmins = 5
                        print(e)
                        print("Rate limit reached for all connections. Waiting %d mins..." % nmins)
                        time.sleep(60 * nmins)
                        conns_tried = 0 # restart count
                    else:
                        self.get_fresh_connection()

    def get_user(self, screen_name=None):
        return self.conn_.get_user(screen_name=screen_name)

API_HANDLER = APIHandler(AUTH_DATA)
