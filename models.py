
from sklearn.linear_model import LogisticRegression


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_squared_error, r2_score
)
import pandas as pd


import pandas as pd
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.metrics import mean_squared_error, r2_score


classifications_models = {

    "LogisticRegression": LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight=None,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        weights="uniform",
        metric="minkowski",
        p=2
    ),

    "SVM": SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        probability=True,
        class_weight=None,
        random_state=42
    ),

    "DecisionTree": DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight=None,
        random_state=42
    ),

    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        subsample=1.0,
        random_state=42
    ),

    "AdaBoost": AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    ),

    "GaussianNB": GaussianNB(
        var_smoothing=1e-9
    ),

    "MultinomialNB": MultinomialNB(
        alpha=1.0
    ),

    "BernoulliNB": BernoulliNB(
        alpha=1.0,
        binarize=0.0
    ),

    "LDA": LinearDiscriminantAnalysis(
        solver="svd"
    )
}
regression_models = {

    "LinearRegression": LinearRegression(
        fit_intercept=True
    ),

    "Ridge": Ridge(
        alpha=1.0,
        solver="auto",
        random_state=42
    ),

    "Lasso": Lasso(
        alpha=0.1,
        max_iter=1000,
        random_state=42
    ),

    "ElasticNet": ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        max_iter=1000,
        random_state=42
    ),

    "KNN": KNeighborsRegressor(
        n_neighbors=5,
        weights="uniform",
        metric="minkowski",
        p=2
    ),

    "SVR": SVR(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        epsilon=0.1
    ),

    "DecisionTree": DecisionTreeRegressor(
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    ),

    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42
    ),

    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        subsample=1.0,
        random_state=42
    ),

    "AdaBoost": AdaBoostRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}
#classification_models لوب لل 
def run_classification_models(models_dict, X_train, X_test, y_train, y_test):
    results = []

    for name, model in models_dict.items():
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test  = model.predict(X_test)

        results.append({
            "model": name,
            "train_acc": accuracy_score(y_train, y_pred_train),
            "test_acc": accuracy_score(y_test, y_pred_test),
            "train_f1": f1_score(y_train, y_pred_train, average="weighted"),
            "test_f1": f1_score(y_test, y_pred_test, average="weighted")
        })

    return pd.DataFrame(results).sort_values("test_f1", ascending=False)

#linear_model لوب لل 

def run_regression_models(models_dict, X_train, X_test, y_train, y_test):
    results = []

    for name, model in models_dict.items():
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test  = model.predict(X_test)

        results.append({
            "model": name,
            "train_mse": mean_squared_error(y_train, y_pred_train),
            "test_mse": mean_squared_error(y_test, y_pred_test),
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": r2_score(y_test, y_pred_test)
        })

    return pd.DataFrame(results).sort_values("test_r2", ascending=False)

