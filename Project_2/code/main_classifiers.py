from ola_edo.dataset_functions import DatasetLoader
from ola_edo.classifier import Classifier

  
# Load dataset
loader = DatasetLoader()

# preprocess = [use_normalization, use_rotation, use_PCA]
preprocess = [1, 1, 0.95]
img_train, label_train, img_test, label_test,  _ = loader.load_cifar(preprocess)


# Filter the dataset for two classes (e.g., class 0 and class 1)
num_classes = 2
img_train, label_train = loader.filter_classes(img_train, label_train, num_classes)
img_test, label_test = loader.filter_classes(img_test, label_test, num_classes)


# Initialize classifiers
knn1 = Classifier("knn", neighbors=1)
knn3 = Classifier("knn", neighbors=3)
centroid = Classifier("centroid")


# Train and evaluate classifiers
# For k-NN, k = 1
knn1.train(img_train, label_train)
accuracy, label_pred = knn1.evaluate(img_test, label_test)
if preprocess[0] == 0 and preprocess[1] == 0:
    print(f"\nk-NN (k=1) Accuracy: {accuracy * 100:.2f}%")
elif preprocess[0] == 1 and preprocess[1] == 0:
    print(f"\nk-NN (k=1) Accuracy Normalized: {accuracy * 100:.2f}%")
elif preprocess[0] == 1 and preprocess[1] == 1:
    print(f"\nk-NN (k=1) Accuracy Normalized (Augmented): {accuracy * 100:.2f}%")
elif preprocess[0] == 0 and preprocess[1] == 1:
    print(f"\nk-NN (k=1) Accuracy (Augmented): {accuracy * 100:.2f}%")
   
# For k-NN, k = 3
knn3.train(img_train, label_train)
accuracy, label_pred = knn3.evaluate(img_test, label_test)
if preprocess[0] == 0 and preprocess[1] == 0:
    print(f"k-NN (k=3) Accuracy: {accuracy * 100:.2f}%")
elif preprocess[0] == 1 and preprocess[1] == 0:
    print(f"k-NN (k=3) Accuracy Normalized: {accuracy * 100:.2f}%")
elif preprocess[0] == 1 and preprocess[1] == 1:
    print(f"k-NN (k=3) Accuracy Normalized (Augmented): {accuracy * 100:.2f}%")
elif preprocess[0] == 0 and preprocess[1] == 1:
    print(f"k-NN (k=3) Accuracy (Augmented): {accuracy * 100:.2f}%")


# For centroid
centroid.train(img_train, label_train)
accuracy, label_pred = centroid.evaluate(img_test, label_test)
if preprocess[0] == 0 and preprocess[1] == 0:
    print(f"Nearest Centroid Accuracy: {accuracy * 100:.2f}%")
elif preprocess[0] == 1 and preprocess[1] == 0:
    print(f"Nearest Centroid Accuracy Normalized: {accuracy * 100:.2f}%")
elif preprocess[0] == 1 and preprocess[1] == 1:
    print(f"Nearest Centroid Accuracy Normalized (Augmented): {accuracy * 100:.2f}%")
elif preprocess[0] == 0 and preprocess[1] == 1:
    print(f"Nearest Centroid Accuracy (Augmented): {accuracy * 100:.2f}%")


