from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.tree import DecisionTreeClassifier

from config.constants import SEED

CLASSIFIERS = [
    LogisticRegression(solver="lbfgs", max_iter=1000, random_state=SEED),
    RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED),
    GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=SEED
    ),
    AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=SEED),
    DecisionTreeClassifier(max_depth=5, random_state=SEED),
    KNeighborsClassifier(n_neighbors=5),
    SVC(kernel="rbf", probability=True, C=1.0, gamma="scale", random_state=SEED),
    GaussianNB(),
    MLPClassifier(hidden_layer_sizes=100, max_iter=300, random_state=SEED),
    HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.1, max_depth=6, random_state=SEED
    ),
]

ONE_CLASS_CLASSIFIERS = [
    IsolationForest(random_state=SEED),
    OneClassSVM(gamma="auto"),
    LocalOutlierFactor(novelty=True),
]
