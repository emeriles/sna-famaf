import json
import os
import numpy as np
from multiprocessing import Manager, Pool

from sklearn.metrics import f1_score, precision_score, recall_score

from modeling._1_one_user_learn_neighbours.model import OneUserModel
from processing._1_user_model.db_csv import DatasetOneUserModel
from processing.utils import get_test_users_ids
from settings import SCORES_FOLDER_1_


def worker(uid, f1s_train, f1s_valid, f1s_testv, precisions_train, precisions_valid, precisions_testv,
           recalls_train, recalls_valid, recalls_testv, pos_cases_train, pos_cases_valid, pos_cases_testv,
           lock, delta_minutes):
    """worker function"""
    print("Largamos para {}".format(uid))
    
    X_train, X_valid, X_testv, y_train, y_valid, y_testv = DatasetOneUserModel.\
                                                        load_or_create_dataset(uid, delta_minutes_filter=delta_minutes)
    clf = OneUserModel.load_or_build_model(uid, 'svc', delta_minutes)

    y_true, y_pred = y_train, clf.predict(X_train)
    lock.acquire()
    f1s_train[uid] = f1_score(y_true, y_pred)
    precisions_train[uid] = precision_score(y_true, y_pred)
    recalls_train[uid] = recall_score(y_true, y_pred)
    pos_cases_train[uid] = int(np.sum(y_true))
    lock.release()

    y_true, y_pred = y_valid, clf.predict(X_valid)
    lock.acquire()
    f1s_valid[uid] = f1_score(y_true, y_pred)
    precisions_valid[uid] = precision_score(y_true, y_pred)
    recalls_valid[uid] = recall_score(y_true, y_pred)
    pos_cases_valid[uid] = int(np.sum(y_true))
    lock.release()

    y_true, y_pred = y_testv, clf.predict(X_testv)
    lock.acquire()
    f1s_testv[uid] = f1_score(y_true, y_pred)
    precisions_testv[uid] = precision_score(y_true, y_pred)
    recalls_testv[uid] = recall_score(y_true, y_pred)
    pos_cases_testv[uid] = int(np.sum(y_true))
    lock.release()


def compute_scores(delta_minutes):

    uids = [u for u in get_test_users_ids()]

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
                                           lock, delta_minutes)
        except ValueError as e:
            print('FAILED FOR USER {}. Exception: {}'.format(uid, e))
    pool.close()
    pool.join()

    if not os.path.exists(SCORES_FOLDER_1_):
        os.makedirs(SCORES_FOLDER_1_)

    with open(SCORES_FOLDER_1_ + '/f1s_train_svc.json', 'w') as f:
        json.dump(dict(f1s_train), f)

    with open(SCORES_FOLDER_1_ + '/f1s_valid_svc.json', 'w') as f:
        json.dump(dict(f1s_valid), f)

    with open(SCORES_FOLDER_1_ + '/f1s_testv_svc.json', 'w') as f:
        json.dump(dict(f1s_testv), f)

    with open(SCORES_FOLDER_1_ + '/precisions_train_svc.json', 'w') as f:
        json.dump(dict(precisions_train), f)

    with open(SCORES_FOLDER_1_ + '/precisions_valid_svc.json', 'w') as f:
        json.dump(dict(precisions_valid), f)

    with open(SCORES_FOLDER_1_ + '/precisions_testv_svc.json', 'w') as f:
        json.dump(dict(precisions_testv), f)

    with open(SCORES_FOLDER_1_ + '/recalls_train_svc.json', 'w') as f:
        json.dump(dict(recalls_train), f)

    with open(SCORES_FOLDER_1_ + '/recalls_valid_svc.json', 'w') as f:
        json.dump(dict(recalls_valid), f)

    with open(SCORES_FOLDER_1_ + '/recalls_testv_svc.json', 'w') as f:
        json.dump(dict(recalls_testv), f)

    with open(SCORES_FOLDER_1_ + '/pos_cases_train_svc.json', 'w') as f:
        json.dump(dict(pos_cases_train), f)

    with open(SCORES_FOLDER_1_ + '/pos_cases_valid_svc.json', 'w') as f:
        json.dump(dict(pos_cases_valid), f)

    with open(SCORES_FOLDER_1_ + '/pos_cases_testv_svc.json', 'w') as f:
        json.dump(dict(pos_cases_testv), f)
