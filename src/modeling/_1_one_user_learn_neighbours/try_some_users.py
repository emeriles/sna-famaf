from modeling._1_one_user_learn_neighbours.model import OneUserModel
from processing._1_user_model.db_csv import DatasetOneUserModel
from processing.utils import get_test_users_ids, get_test_users_ids_aux


def worker(user_id, delta_minutes):
    # try:
        # clf = train_and_evaluate(user_id)
    X_train, X_valid, X_test, y_train, y_valid, y_test = DatasetOneUserModel\
        .load_or_create_dataset(user_id, delta_minutes_filter=delta_minutes)
    dataset = X_train, X_valid, y_train, y_valid
    clf = OneUserModel.model_select_svc(dataset)
    OneUserModel.save_model(clf, user_id, 'svc', time_delta_filter=delta_minutes)
    # except Exception as e:
    #     print(e)
    #     print(e)
    #     print(e)


def try_some_users(delta_minutes, cherry_pick_users=False):
    # See which users are pending
    # pending_user_ids = []
    # for user_id, username, _ in TEST_USERS_ALL:
    #     try:
    #         load_model_small(user_id, 'svc')
    #     except IOError:
    #         pending_user_ids.append(user_id)


    # pool = Pool(processes=6)
    # for user_id in pending_user_ids:
    #     pool.apply_async(worker, (user_id,))
    # pool.close()
    # pool.join()

    # pending_user_ids = [uid for uid,_,_ in TEST_USERS_ALL]

    print('Using cherry picked users: {}'.format(cherry_pick_users))
    if cherry_pick_users:
        uids = get_test_users_ids_aux()
    else:
        uids = get_test_users_ids()

    for user_id in uids:
            # [#74153376, 1622441, 117335842,
            #         76133133, 33524608, 85861402]:
        print('Try some users for user {}'.format(user_id))
        try:
            worker(user_id, delta_minutes=delta_minutes)
        except ValueError as e:
            print(e)
            raise e
