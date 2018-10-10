import pprint

import time

from pymongo import MongoClient, InsertOne
from pymongo.errors import BulkWriteError

from twitter_api import API_HANDLER


class DBHandler(object):

    def __init__(self):
        self._host = 'localhost'
        self._port = 27017
        self._db_name = 'tweets'
        self.api_handler = API_HANDLER

        self.db = MongoClient(self._host, self._port).twitter
        self.tweet_collection = self.db.tweet
        self.users_ids_collection = self.db.user_id

        # set tweet collection's primary key as 'id'
        self.tweet_collection.create_index('id', unique=True)

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

    def save_users_timelines(self, desde=None, hasta=None, dia=None, limite=None):
        start_time = time.time()
        print("Saving user timeline: desde: {desde}\thasta: {hasta}\t dia: {dia}\t limite: {limite}".format(
            desde=desde, hasta=hasta, dia=dia, limite=limite
        ))
        users_ids = self.get_users_ids()

        to_process = len(users_ids)
        percentage = 0

        for u_id in users_ids:
            # advance notification
            to_process -= 1
            percentage = 100 - int(to_process / len(users_ids) * 100)
            print(
                'Avance: %{}; {}/{}\t\t\tProcesando para guardar timeline del usuario {}'.format(percentage, to_process, len(users_ids), u_id)
            )

            self.save_timeline_for_user(user_id=u_id)
        elapsed_time = time.time() - start_time
        print("Done saving users timelines. Took: {}.1f secs".format(elapsed_time))
        return 0

    def save_users_ids(self, user_ids=[]):
        # convert to dict
        # _id is default primary key, so it will be overridden.
        d_ids = [{'_id': x} for x in user_ids]
        self.users_ids_collection.insert_many(d_ids)
        return 0

    def get_users_ids(self):
        as_list = [x.get('_id', None) for x in self.users_ids_collection.find()]
        return as_list
