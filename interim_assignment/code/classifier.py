from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import accuracy_score

class Classifier:
    def __init__(self, classifier_type="knn", neighbors=1):
        if classifier_type == "knn":
            self.clf = KNeighborsClassifier(n_neighbors=neighbors)
        elif classifier_type == "centroid":
            self.clf = NearestCentroid()

    def train(self, img_train, label_train):
        self.clf.fit(img_train, label_train)

    def evaluate(self, img_test, label_test):
        label_pred = self.clf.predict(img_test)
        accuracy = accuracy_score(label_test, label_pred)
        return accuracy, label_pred
