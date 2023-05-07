from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm, naive_bayes
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load

def create_model() -> Pipeline:
    return knn_model()

def svm_model() -> Pipeline:
    svc = svm.SVC(C=1, kernel="rbf", gamma=0.1, random_state=42, probability=True)
    clf = make_pipeline(StandardScaler(), svc)
    return clf 

def bayes_model() -> Pipeline:
    svc = naive_bayes.ComplementNB()
    clf = make_pipeline(StandardScaler(), svc)
    return clf 

def knn_model() -> Pipeline:
    svc = KNeighborsClassifier(n_neighbors=9)
    clf = make_pipeline(StandardScaler(), svc)
    return clf 

def save_model(clf: Pipeline, filename: str):
    dump(clf, filename) 

def load_model(filename: str) -> Pipeline:
    return load(filename) 