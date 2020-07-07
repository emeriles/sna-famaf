from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

target_names = ["Insignificante", "Relevante"]

def SVM_gridsearch(x_train, x_test, y_train, y_true, cv=3):
    parameters = [
        {
         'kernel': ['rbf', 'poly', 'linear'],
         'gamma': [0.66, 0.77, 0.88, 0.99, 1.5],
         'C': [0.1, 0.5, 0.8],
         'class_weight': ['balanced', None]
        }
    ]

    scores = [
        # 'precision',
        # 'recall',
        'f1'
    ]
    from sklearn import svm
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        clf = GridSearchCV(
            svm.SVC(),
            param_grid=parameters,
            refit=True,  # fit using all data, on the best detected classifier
            n_jobs=-1,
            scoring=score,
            cv=cv,
        )

        clf.fit(x_train, y_train)

        print("Best parameters set found on training set:")
        print("")
        print(clf.best_params_)

        print("Detailed classification report:")
        print("")
        print("Scores on training set.")
        y_pred = clf.predict(x_train)
        print(classification_report(y_train, y_pred, digits=4))
        print()
        y_pred = clf.predict(x_test)
        print("")
        print("Results on test dataset with SVM")
        print("")
        print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
        print("")
    return clf
