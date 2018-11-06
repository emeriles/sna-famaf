import os
import pickle
import pprint
import pymongo

import time

from pymongo import MongoClient, InsertOne
from pymongo.errors import BulkWriteError


TMP_SEEN_USERS = './temp/seen_users.pickle'

SEEN_USERS = set()
if os.path.exists(TMP_SEEN_USERS):
    with open(TMP_SEEN_USERS, 'rb') as f:
        SEEN_USERS = pickle.load(f)


class DBHandler(object):

    def __init__(self, port=27017):
        from twitter_api import API_HANDLER
        self._host = 'localhost'
        self._port = port
        self._db_name = 'tweets'
        self.api_handler = API_HANDLER

        print('conecting to db')
        self.db = MongoClient(self._host, self._port).twitter
        self.tweet_collection = self.db.tweet
        self.users_ids_collection = self.db.user_id

        # set tweet collection's primary key as 'id'. this corrupts bd! 'id' not unique in twitter?
        # very dificul
        # self.tweet_collection.create_index('id', unique=True)

    def save_timeline_for_user(self, user_id, desde=None, hasta=None, dia=None, limite=None):

        print("Fetching timeline for user %s" % user_id)
        start_time = time.time()
        # authenticating here ensures a different set of credentials
        # everytime we start processing a new county, to prevent hitting the rate limit
        self.api_handler.get_fresh_connection()

        tweets = self.api_handler.traer_timeline(user_id=user_id, desde=desde, hasta=hasta, dia=dia, limite=limite)

        request = [InsertOne(x) for x in tweets]
        save_status = None
        try:
            # if not empty tweets save them
            if request:
                save_status = self.tweet_collection.bulk_write(request, ordered=False)
        except BulkWriteError as e:
            write_errors = e.details['writeErrors']
            # duplicates = [x for x in write_errors if 'duplicate key error' in x['errmsg']]
            non_duplicates = [x for x in write_errors if 'duplicate key error' not in x['errmsg']]

            print('Errors with non duplicates: {}. '.format(len(non_duplicates)))
            pprint.pprint(set([x['errmsg'] for x in non_duplicates]))
            print('End errors with non duplicates.')

        elapsed_time = time.time() - start_time
        print("Done fetch for user {}. Took {}.1f secs to fetch {} tweets" .format(user_id, elapsed_time, len(tweets)))
        print('New saved tweets: {}'.format(save_status.inserted_count if save_status is not None else 0))
        return 0

    def save_users_timelines(self, desde=None, hasta=None, dia=None, limite=None, use_milestones=False):
        start_time = time.time()
        print("Saving user timeline: desde: {desde}\thasta: {hasta}\t dia: {dia}\t limite: {limite}".format(
            desde=desde, hasta=hasta, dia=dia, limite=limite
        ))
        users_ids = set(self.get_users_ids())

        if use_milestones:
            users_ids = users_ids.difference(SEEN_USERS)

        to_process = len(users_ids)

        for u_id in users_ids:
            # advance notification
            to_process -= 1
            percentage = 100 - int(to_process / len(users_ids) * 100)
            print(
                'Avance: %{}; {}/{}\t\t\tProcesando para guardar timeline del usuario {}'.format(percentage, to_process, len(users_ids), u_id)
            )

            self.save_timeline_for_user(user_id=u_id, desde=desde, hasta=hasta, dia=dia, limite=limite)

            SEEN_USERS.add(u_id)
            with open(TMP_SEEN_USERS, 'wb') as f:
                pickle.dump(SEEN_USERS, f)

        elapsed_time = time.time() - start_time
        print("Done saving users timelines. Took: {}.1f secs".format(elapsed_time))
        return 0

    def save_users_ids(self, user_ids=[]):
        # convert to dict
        # _id is default primary key, so it will be overridden.
        d_ids = [{'_id': x} for x in user_ids]
        request = [InsertOne(x) for x in d_ids]
        try:
            # if not empty tweets save them
            if request:
                save_status = self.users_ids_collection.bulk_write(request, ordered=False)
        except BulkWriteError as e:
            # TODO: extract this to function?
            write_errors = e.details['writeErrors']
            # duplicates = [x for x in write_errors if 'duplicate key error' in x['errmsg']]
            non_duplicates = [x for x in write_errors if 'duplicate key error' not in x['errmsg']]

            print('Errors with non duplicates: {}. '.format(len(non_duplicates)))
            pprint.pprint(set([x['errmsg'] for x in non_duplicates]))
            print('End errors with non duplicates.')
        return 0

    def get_users_ids(self):
        as_list = [x.get('_id', None) for x in self.users_ids_collection.find()]
        return as_list

    def get_min_max_values(self, field):
        """
        Returns min and max values for given field.

        :param field:
        :return:
        """
        cursor = self.tweet_collection.aggregate([
            { '$group': {
                '_id': None,
                'max': {'$max': '$' + field},
                'min': {'$min': '$' + field},
            }
            }
        ])
        return list(cursor)[0]

    def get_n_max_registry(self, field_name, n=1):
        """
        Returns `n` max tweets for a given field
        :param field_name:
        :param n:
        :return:
        """
        return list(self.tweet_collection.find().sort([
            (field_name, pymongo.DESCENDING)
        ]).limit(n))

    def query_tweets(self, filters=None, select_fields=None, only_retweets=False):
        """
        Simple abstraction of tweet query.
        Example:
         `h.query_tweets(only_retweets=True, select_fields=['created_at', 'retweeted_status.created_at'])`

        :param filters:
        :param select_fields:
        :param only_retweets:
        :return:
        """
        # set up filters
        if filters and not isinstance(filters, dict):
            raise Exception('Argument `filters` should be a dict!')
        if not filters:
            filters = {}
        if only_retweets:
            rt_status = filters.get('retweeted_status', None)
            if not rt_status:
                filters.update({'retweeted_status':
                                {'$ne': None}
                                })
            else:
                # TODO: check this. not full support for filters...
                rt_status.update({'$ne': None})

        # set up projections
        projection = {}
        projection.update({key: 1 for key in select_fields} if projection is not None else {})

        cursor = self.tweet_collection.find(filter=filters, projection=projection)
        return cursor
