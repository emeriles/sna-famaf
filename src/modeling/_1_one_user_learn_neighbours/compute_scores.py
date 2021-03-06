import json
import os
import numpy as np
from multiprocessing import Manager, Pool

from sklearn.metrics import f1_score, precision_score, recall_score

from modeling._1_one_user_learn_neighbours.model import OneUserModel
from processing._1_user_model.db_csv import DatasetOneUserModel
from processing.utils import get_test_users_ids, get_test_users_ids_aux
from settings import SCORES_FOLDER_1_, SCORES_FOLDER_1_FT


def worker(uid, f1s_train, f1s_valid, f1s_testv, precisions_train, precisions_valid, precisions_testv,
           recalls_train, recalls_valid, recalls_testv, pos_cases_train, pos_cases_valid, pos_cases_testv,
           lock, delta_minutes, fasttext, as_seconds=False):
    """worker function"""
    print("Largamos para {}".format(uid))

    X_train, X_valid, X_testv, y_train, y_valid, y_testv, X_train_l, X_test_l, X_valid_l = DatasetOneUserModel.\
                                                        load_or_create_dataset(uid, delta_minutes_filter=delta_minutes,
                                                                               fasttext=fasttext, as_seconds=as_seconds)
    # old loading
    # X_train, X_valid, X_testv, y_train, y_valid, y_testv = DatasetOneUserModel.load_or_create_dataset(central_uid, delta_minutes_filter=0)
    clf = OneUserModel.load_or_build_model(uid, 'svc', delta_minutes, fasttext=fasttext, as_seconds=as_seconds)

    y_true, y_pred = y_train, clf.predict(X_train)
    lock.acquire()
    f1s_train[uid] = f1_score(y_true, y_pred)
    precisions_train[uid] = precision_score(y_true, y_pred)
    recalls_train[uid] = recall_score(y_true, y_pred)
    pos_cases_train[uid] = int(np.sum(y_true))
    lock.release()

    # y_true, y_pred = y_valid, clf.predict(X_valid)
    # lock.acquire()
    # f1s_valid[uid] = f1_score(y_true, y_pred)
    # precisions_valid[uid] = precision_score(y_true, y_pred)
    # recalls_valid[uid] = recall_score(y_true, y_pred)
    # pos_cases_valid[uid] = int(np.sum(y_true))
    # lock.release()

    y_true, y_pred = y_testv, clf.predict(X_testv)
    lock.acquire()
    f1s_testv[uid] = f1_score(y_true, y_pred)
    precisions_testv[uid] = precision_score(y_true, y_pred)
    recalls_testv[uid] = recall_score(y_true, y_pred)
    pos_cases_testv[uid] = int(np.sum(y_true))
    lock.release()


def compute_scores(delta_minutes, cherry_pick_users=False, fasttext=False, as_seconds=False):

    print('Using cherry picked users: {}'.format(cherry_pick_users))
    if cherry_pick_users:
        uids = get_test_users_ids_aux()
    else:
        uids = get_test_users_ids()

    print('Running compute scores for {} users.'.format(len(uids)))

    pool = Pool(processes=7)

    manager = Manager()

    f1s_train = manager.dict()
    f1s_valid = manager.dict()
    f1s_testv = manager.dict()

    precisions_train = manager.dict()
    precisions_valid = manager.dict()
    precisions_testv = manager.dict()

    recalls_train = manager.dict()
    recalls_valid = manager.dict()
    recalls_testv = manager.dict()

    pos_cases_train = manager.dict()
    pos_cases_valid = manager.dict()
    pos_cases_testv = manager.dict()

    lock = manager.Lock()

    for uid in uids:
        try:
            worker(uid, f1s_train, f1s_valid, f1s_testv,
                                           precisions_train, precisions_valid, precisions_testv,
                                           recalls_train, recalls_valid, recalls_testv,
                                           pos_cases_train, pos_cases_valid, pos_cases_testv,
                                           lock, delta_minutes, fasttext=fasttext, as_seconds=as_seconds)
        except ValueError as e:
            print('FAILED FOR USER {}. Exception: {}'.format(uid, e))
    pool.close()
    pool.join()

    scores_folder = SCORES_FOLDER_1_FT if fasttext else SCORES_FOLDER_1_
    delta_minutes = str(delta_minutes) + 'secs' if as_seconds else str(delta_minutes)

    if not os.path.exists(scores_folder):
        os.makedirs(scores_folder)

    with open(scores_folder + '/f1s_train_{}_svc.json'.format(delta_minutes), 'w') as f:
        json.dump(dict(f1s_train), f)

    with open(scores_folder + '/f1s_valid_{}_svc.json'.format(delta_minutes), 'w') as f:
        json.dump(dict(f1s_valid), f)

    with open(scores_folder + '/f1s_testv_{}_svc.json'.format(delta_minutes), 'w') as f:
        json.dump(dict(f1s_testv), f)

    with open(scores_folder + '/precisions_train_{}_svc.json'.format(delta_minutes), 'w') as f:
        json.dump(dict(precisions_train), f)

    with open(scores_folder + '/precisions_valid_{}_svc.json'.format(delta_minutes), 'w') as f:
        json.dump(dict(precisions_valid), f)

    with open(scores_folder + '/precisions_testv_{}_svc.json'.format(delta_minutes), 'w') as f:
        json.dump(dict(precisions_testv), f)

    with open(scores_folder + '/recalls_train_{}_svc.json'.format(delta_minutes), 'w') as f:
        json.dump(dict(recalls_train), f)

    with open(scores_folder + '/recalls_valid_{}_svc.json'.format(delta_minutes), 'w') as f:
        json.dump(dict(recalls_valid), f)

    with open(scores_folder + '/recalls_testv_{}_svc.json'.format(delta_minutes), 'w') as f:
        json.dump(dict(recalls_testv), f)

    with open(scores_folder + '/pos_cases_train_{}_svc.json'.format(delta_minutes), 'w') as f:
        json.dump(dict(pos_cases_train), f)

    with open(scores_folder + '/pos_cases_valid_{}_svc.json'.format(delta_minutes), 'w') as f:
        json.dump(dict(pos_cases_valid), f)

    with open(scores_folder + '/pos_cases_testv_{}_svc.json'.format(delta_minutes), 'w') as f:
        json.dump(dict(pos_cases_testv), f)
