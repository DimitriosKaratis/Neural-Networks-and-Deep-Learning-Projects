from dataset_functions import DatasetLoader
from classifier import Classifier
import time
  
# Load dataset
loader = DatasetLoader()

# preprocess = [use_normalization, use_rotation, use_PCA]
preprocess = [1, 1, 0.0]
img_train, label_train, img_test, label_test,  _ = loader.load_cifar(preprocess)

# Initialize classifiers
knn1 = Classifier("knn", neighbors=1)
knn3 = Classifier("knn", neighbors=3)
centroid = Classifier("centroid")


# Train and evaluate classifiers
# For k-NN, k = 1
start_time = time.time()             
train_acc = knn1.train(img_train, label_train)
end_time = time.time()
elapsed_time = end_time - start_time

accuracy, label_pred = knn1.evaluate(img_test, label_test)
if preprocess[0] == 0 and preprocess[1] == 0:
    print(f"\nk-NN (k=1) Accuracy: {accuracy * 100:.2f}%, train_acc: {train_acc * 100:.2f}%")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
elif preprocess[0] == 1 and preprocess[1] == 0:
    print(f"\nk-NN (k=1) Accuracy Normalized: {accuracy * 100:.2f}%, train_acc: {train_acc * 100:.2f}%")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
elif preprocess[0] == 1 and preprocess[1] == 1:
    print(f"\nk-NN (k=1) Accuracy Normalized (Augmented): {accuracy * 100:.2f}%, train_acc: {train_acc * 100:.2f}%")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
elif preprocess[0] == 0 and preprocess[1] == 1:
    print(f"\nk-NN (k=1) Accuracy (Augmented): {accuracy * 100:.2f}%, train_acc: {train_acc * 100:.2f}%")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")



# For k-NN, k = 3
start_time = time.time()   
train_acc = knn3.train(img_train, label_train)
end_time = time.time()
elapsed_time = end_time - start_time

accuracy, label_pred = knn3.evaluate(img_test, label_test)
if preprocess[0] == 0 and preprocess[1] == 0:
    print(f"k-NN (k=3) Accuracy: {accuracy * 100:.2f}%, train_acc: {train_acc * 100:.2f}%")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
elif preprocess[0] == 1 and preprocess[1] == 0:
    print(f"k-NN (k=3) Accuracy Normalized: {accuracy * 100:.2f}%, train_acc: {train_acc * 100:.2f}%")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
elif preprocess[0] == 1 and preprocess[1] == 1:
    print(f"k-NN (k=3) Accuracy Normalized (Augmented): {accuracy * 100:.2f}%, train_acc: {train_acc * 100:.2f}%")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
elif preprocess[0] == 0 and preprocess[1] == 1:
    print(f"k-NN (k=3) Accuracy (Augmented): {accuracy * 100:.2f}%, train_acc: {train_acc * 100:.2f}%")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")

# For centroid
start_time = time.time()   
train_acc = centroid.train(img_train, label_train)
end_time = time.time()
elapsed_time = end_time - start_time

accuracy, label_pred = centroid.evaluate(img_test, label_test)
if preprocess[0] == 0 and preprocess[1] == 0:
    print(f"Nearest Centroid Accuracy: {accuracy * 100:.2f}%, train_acc: {train_acc * 100:.2f}%")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
elif preprocess[0] == 1 and preprocess[1] == 0:
    print(f"Nearest Centroid Accuracy Normalized: {accuracy * 100:.2f}%, train_acc: {train_acc * 100:.2f}%")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
elif preprocess[0] == 1 and preprocess[1] == 1:
    print(f"Nearest Centroid Accuracy Normalized (Augmented): {accuracy * 100:.2f}%, train_acc: {train_acc * 100:.2f}%")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
elif preprocess[0] == 0 and preprocess[1] == 1:
    print(f"Nearest Centroid Accuracy (Augmented): {accuracy * 100:.2f}%, train_acc: {train_acc * 100:.2f}%")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")