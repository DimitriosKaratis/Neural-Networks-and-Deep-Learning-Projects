from dataset_functions import DatasetLoader
from classifier import Classifier
from visualization import Visualization
from paths_config import Config
  
# Load dataset
loader = DatasetLoader()
cifar_dir = Config.DATASET_PATH 

# preprocess = [use_normalization, use_rotation, use_validation, use_PCA]
preprocess = [1, 1, 0, 0]
img_train, label_train, img_test, label_test, img_val, label_val, _, _, _, _ = loader.load_cifar10_data(cifar_dir, preprocess)

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
    #Visualization.generate_conf_matrix(label_test, label_pred, "Confusion Matrix for k-NN (k=1)")
elif preprocess[0] == 1 and preprocess[1] == 0:
    print(f"\nk-NN (k=1) Accuracy Normalized: {accuracy * 100:.2f}%")
    #Visualization.generate_conf_matrix(label_test, label_pred, "Confusion Matrix for k-NN (k=1) Normalized")
elif preprocess[0] == 1 and preprocess[1] == 1:
    print(f"\nk-NN (k=1) Accuracy Normalized (Augmented): {accuracy * 100:.2f}%")
    #Visualization.generate_conf_matrix(label_test, label_pred, "Confusion Matrix for k-NN (k=1) Normalized (Augmented)")
elif preprocess[0] == 0 and preprocess[1] == 1:
    print(f"\nk-NN (k=1) Accuracy (Augmented): {accuracy * 100:.2f}%")
    #Visualization.generate_conf_matrix(label_test, label_pred, "Confusion Matrix for k-NN (k=1) (Augmented)")


# For k-NN, k = 3
knn3.train(img_train, label_train)
accuracy, label_pred = knn3.evaluate(img_test, label_test)
if preprocess[0] == 0 and preprocess[1] == 0:
    print(f"k-NN (k=3) Accuracy: {accuracy * 100:.2f}%")
    #Visualization.generate_conf_matrix(label_test, label_pred, "Confusion Matrix for k-NN (k=3)")
elif preprocess[0] == 1 and preprocess[1] == 0:
    print(f"k-NN (k=3) Accuracy Normalized: {accuracy * 100:.2f}%")
    #Visualization.generate_conf_matrix(label_test, label_pred, "Confusion Matrix for k-NN (k=3) Normalized")
elif preprocess[0] == 1 and preprocess[1] == 1:
    print(f"k-NN (k=3) Accuracy Normalized (Augmented): {accuracy * 100:.2f}%")
    #Visualization.generate_conf_matrix(label_test, label_pred, "Confusion Matrix for k-NN (k=3) Normalized (Augmented)")
elif preprocess[0] == 0 and preprocess[1] == 1:
    print(f"k-NN (k=3) Accuracy (Augmented): {accuracy * 100:.2f}%")
    #Visualization.generate_conf_matrix(label_test, label_pred, "Confusion Matrix for k-NN (k=3) (Augmented)")


# For centroid
centroid.train(img_train, label_train)
accuracy, label_pred = centroid.evaluate(img_test, label_test)
if preprocess[0] == 0 and preprocess[1] == 0:
    print(f"Nearest Centroid Accuracy: {accuracy * 100:.2f}%")
    #Visualization.generate_conf_matrix(label_test, label_pred, "Confusion Matrix for Nearest Centroid Classifier")
elif preprocess[0] == 1 and preprocess[1] == 0:
    print(f"Nearest Centroid Accuracy Normalized: {accuracy * 100:.2f}%")
    #Visualization.generate_conf_matrix(label_test, label_pred, "Confusion Matrix for Nearest Centroid Classifier Normalized")
elif preprocess[0] == 1 and preprocess[1] == 1:
    print(f"Nearest Centroid Accuracy Normalized (Augmented): {accuracy * 100:.2f}%")
    #Visualization.generate_conf_matrix(label_test, label_pred, "Confusion Matrix for Nearest Centroid Classifier Normalized (Augmented)")
elif preprocess[0] == 0 and preprocess[1] == 1:
    print(f"Nearest Centroid Accuracy (Augmented): {accuracy * 100:.2f}%")
    #Visualization.generate_conf_matrix(label_test, label_pred, "Confusion Matrix for Nearest Centroid Classifier (Augmented)")
