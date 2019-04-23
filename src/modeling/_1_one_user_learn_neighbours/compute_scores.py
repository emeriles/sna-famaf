from multiprocessing import Manager, Pool
from sklearn.metrics import f1_score, precision_score, recall_score
import json

from modeling._1_one_user_learn_neighbours.model import OneUserModel
from processing.datasets import load_or_create_dataframe_validation
from settings import TEST_USERS_ALL, SCORES_FOLDER_1_


def worker(uid, f1s_train, f1s_valid, f1s_testv, precisions_train, precisions_valid, precisions_testv,
    recalls_train, recalls_valid, recalls_testv, lock):
    """worker function"""
    print("Largamos para %d" % uid)
    
    X_train, X_valid, X_testv, y_train, y_valid, y_testv = load_or_create_dataframe_validation(uid)
    clf = OneUserModel.load_or_build_model(uid, 'svc')
    print('datafff', X_train, X_valid, X_testv, y_train, y_valid, y_testv)

    y_true, y_pred = y_train, clf.predict(X_train)
    lock.acquire()
    f1s_train[uid] = f1_score(y_true, y_pred)
    precisions_train[uid] = precision_score(y_true, y_pred)
    recalls_train[uid] = recall_score(y_true, y_pred)
    lock.release()

    y_true, y_pred = y_valid, clf.predict(X_valid)
    lock.acquire()
    f1s_valid[uid] = f1_score(y_true, y_pred)
    precisions_valid[uid] = precision_score(y_true, y_pred)
    recalls_valid[uid] = recall_score(y_true, y_pred) 
    lock.release()

    y_true, y_pred = y_testv, clf.predict(X_testv)
    lock.acquire()
    f1s_testv[uid] = f1_score(y_true, y_pred)
    precisions_testv[uid] = precision_score(y_true, y_pred)
    recalls_testv[uid] = recall_score(y_true, y_pred)
    lock.release()


def compute_scores():

    uids = [u[0] for u in TEST_USERS_ALL]

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

    lock = manager.Lock()

    for uid in uids:
        worker(uid, f1s_train, f1s_valid, f1s_testv,
                                       precisions_train, precisions_valid, precisions_testv,
                                       recalls_train, recalls_valid, recalls_testv,
                                       lock)
    pool.close()
    pool.join()

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
