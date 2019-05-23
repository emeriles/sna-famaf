from modeling._1_one_user_learn_neighbours.model import OneUserModel
from processing.db_csv import Dataset


def worker(user_id, delta_minutes):
    # try:
        # clf = train_and_evaluate(user_id)
    X_train, X_valid, X_test, y_train, y_valid, y_test = Dataset.load_or_create_dataset(user_id, delta_minutes_filter=delta_minutes)
    dataset = X_train, X_valid, y_train, y_valid
    clf = OneUserModel.model_select_svc(dataset)
    OneUserModel.save_model(clf, user_id, 'svc', time_delta_filter=delta_minutes)
    # except Exception as e:
    #     print(e)
    #     print(e)
    #     print(e)


def try_some_users(delta_minutes):
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

    for user_id in [74153376, 1622441, 117335842]:
        print('Try some users for user {}'.format(user_id))
        worker(user_id, delta_minutes=delta_minutes)
