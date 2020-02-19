import pickle
import pprint
from collections import defaultdict

from sklearn.metrics import classification_report, roc_curve, roc_auc_score, precision_score

from processing._influencers_model.grid_searchers import SVM_gridsearch
from settings import C, GAMMA, EVAL_TRAIN, GRIDSEARCH, MIN_INFLUENCERS, MAX_INFLUENCERS, STEP_INFLUENCERS, \
    XY_CACHE_FOLDER

pp = pprint.PrettyPrinter(indent=2)
target_names = ["Insignificante", "Relevante"]


def svm_classifier(x_train, x_test, y_train, y_true, gridsearch=False,
                   file=None, roc=False, roc_curve_=False):
    from sklearn import svm
    if gridsearch:
        clf = SVM_gridsearch(x_train, x_test, y_train, y_true)
    else:
        clf = svm.SVC(kernel='rbf', C=C, gamma=GAMMA, class_weight='balanced', shrinking=True, probability=True)
        #clf = svm.SVC(kernel='poly', C=0.1, gamma=1, class_weight='balanced')
    print('Fitting...')
    clf.fit(x_train, y_train)
    print('Predicting...')
    y_pred = clf.predict(x_test)
    y_scores = clf.predict_proba(x_test)[:,1]
    if roc_curve_:
        return roc_curve(y_true, y_scores)
    if roc:
        return roc_auc_score(y_true, y_scores)
    print("")
    print("Results on {} dataset with SVM as:".format(file))
    print("")
    pp.pprint(clf.get_params())
    print("")
    print(classification_report(y_true, y_pred, target_names=target_names))
    print("")
    return precision_score(y_true, y_pred, pos_label=1)
    return f1_score(y_true, y_pred, pos_label=1)


def results_on_experiment(folder, file_or_files, roc=False, roc_curve_=False):
    def _report_on(file):
        try:
            with open("{}_infl_y_test_{}.pickle".format(folder, file), "rb") as f:
                y_test = pickle.load(f)
            with open("{}_infl_X_test_{}.pickle".format(folder, file), "rb") as f:
                x_test = pickle.load(f)
            with open("{}_infl_y_train_{}.pickle".format(folder, file), "rb") as f:
                y_train = pickle.load(f)
            with open("{}_infl_X_train_{}.pickle".format(folder, file), "rb") as f:
                x_train = pickle.load(f)
        except Exception as e:
            print(str(e))
            raise e
        print("=======================================================")
        print("File: {}".format(file))
        print("(train)Tweets: {}, Influencers: {}".format(x_train.shape[0],
                                                          x_train.shape[1]))
        print("(test) Tweets: {}, Influencers: {}".format(x_test.shape[0],
                                                          x_test.shape[1]))
        print("=======================================================")
        '''
        f1 = svm_classifier(x_train,
                            x_test,
                            y_train,
                            y_test,
                            gridsearch=False,
                            file=file)
        f1_train = svm_classifier(x_train,
                                  x_train,
                                  y_train,
                                  y_train,
                                  gridsearch=False,
                                  file=file)
        '''
        # Another options
        f1 = svm_classifier(x_train, x_test, y_train, y_test, gridsearch=GRIDSEARCH, file=file, roc=roc,
                            roc_curve_=roc_curve_)
        f1_train = f1
        if EVAL_TRAIN:
            f1_train = svm_classifier(x_train, x_train, y_train, y_train, gridsearch=False, file=file, roc=roc,
                                      roc_curve_=roc_curve_)
        # gaussian_truncated_svd(x_train, x_test, y_train, y_true)
        # neigh_classifier(x_train, x_test, y_train, y_true, gridsearch=False)
        return f1, f1_train

    def prom(list):
        return float(sum(list) / len(list))

    if isinstance(file_or_files, list):
        avg_f1 = []
        avg_f1_train = []
        for file in file_or_files:
            f1_score, f1_score_train = _report_on(file)
            avg_f1.append(f1_score)
            avg_f1_train.append(f1_score_train)
        return prom(avg_f1), prom(avg_f1_train)
    else:
        return _report_on(file_or_files)


def f1_score_by_influencers(folders, delta_minutes):
    def get_multiple_tries(file, folder):
        from os import listdir
        from os.path import isfile, join
        onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f)) and\
                     ("train_" + file) in f]
        target = list(set(map(lambda x: (x.split("train_")[-1]
                                         .split("test_")[-1]
                                         .split(".pickle")[0]), onlyfiles)))
        target.sort()
        print(file, target)
        return target

    all_results = []
    suffixes = ["_social",
                # "_tw_lda_60",
                # "_random",
                # "_fasttext",
                # "_tw_lda_10"
                ]
               #"_tw_lda_10", "_tw_lda_20", "_tw_lda_60", "_tw_lda_40",
               # "_tw_lda_50", "_tw_lda_60", "_tw_lda_70", "_tw_lda_80",
               # "_social", "_lda_10", "_lda_20", "_random", "_fasttext"]
    for FOLDER in folders:
        results = []
        testing_files = []
        for idsuffix, suffix in enumerate(suffixes):
            prefixes = range(MIN_INFLUENCERS, MAX_INFLUENCERS, STEP_INFLUENCERS)
            testing_files = ["{}i{}_{}m".format(each, suffix, delta_minutes) for each in prefixes]
            results.append(defaultdict(list))
            results[idsuffix]['x'] = prefixes
            for filename in testing_files:
                try:
                    file = get_multiple_tries(filename, FOLDER)
                    f1, f1_train = results_on_experiment(FOLDER, file)
                    results[idsuffix]['y'].append(f1)
                    if f1 != f1_train:
                        results[idsuffix]['y_train'].append(f1_train)
                except Exception as e:
                    print(e)
                    print("OMMITED:{}".format(filename))
                    raise e
            results[idsuffix]['name'] = suffix
            results[idsuffix]['name'] = "Soc.+" + suffix if "Social" in FOLDER else results[idsuffix]['name']
            results[idsuffix]['name'] = "Random Infl." if "Random++" in FOLDER else results[idsuffix]['name']
        all_results.extend(results)
    # from plot2d import Plot
    # Plot("",#"GAMMA:{} C:{}".format(GAMMA, C),
    #      "number of influencers used for prediction",
    #      "f1-score over positive class",
    #      all_results,
    #      "graph_{}_{}.html".format(GAMMA, C))


def evaluate(delta_minutes):
    f1_score_by_influencers([XY_CACHE_FOLDER], delta_minutes)
