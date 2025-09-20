########################################## INTERIM ASSIGNMENT ###########################################################

# KARATIS DIMITRIOS 10775

from dataset_functions import DatasetLoader
from classifier import Classifier
from visualization import Visualization
  
# Load dataset
loader = DatasetLoader()
cifar_dir = '/home/karatisd/cifar-10/'  

preprocess = [1, 1, 0]
img_train, label_train, img_test, label_test, img_val, label_val = loader.load_cifar10_data(cifar_dir, preprocess)

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
    Visualization.generate_conf_matrix(label_test, label_pred, "Confusion Matrix for k-NN (k=1) Normalized (Augmented)")
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
    Visualization.generate_conf_matrix(label_test, label_pred, "Confusion Matrix for k-NN (k=3) Normalized (Augmented)")
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
    Visualization.generate_conf_matrix(label_test, label_pred, "Confusion Matrix for Nearest Centroid Classifier Normalized (Augmented)")
elif preprocess[0] == 0 and preprocess[1] == 1:
    print(f"Nearest Centroid Accuracy (Augmented): {accuracy * 100:.2f}%")
    #Visualization.generate_conf_matrix(label_test, label_pred, "Confusion Matrix for Nearest Centroid Classifier (Augmented)")


##################################### MY NEURAL NETWORKS ############################################################


# from dataset_functions import DatasetLoader
# from classifier import Classifier
# from MLP import NeuralNetworkMLP
# from CNN import NeuralNetworkCNN
# from visualization import Visualization
# import numpy as np
# import tensorflow as tf
  
# # Load dataset
# loader = DatasetLoader()
# cifar_dir = '/home/karatisd/cifar-10/'  


# # normalize = 1, use_rotation = 1, use validation set = 1
# preprocess = [1, 1, 1]

# #(img_train, label_train), (img_test, label_test) = tf.keras.datasets.cifar10.load_data()
# img_train, label_train, img_test, label_test, img_val, label_val = loader.load_cifar10_data(cifar_dir, preprocess)


# num_of_classes = 10

# MLP = NeuralNetworkMLP(num_of_classes, layer_sizes=[128,64,32], activation_functions=['relu','relu','relu'], dropout_rate=0.1)

# MLP.train(img_train, label_train, img_val, label_val, epochs=200, batch_size=64)

# accuracy, predictions = MLP.evaluate(img_test, label_test)
# print(f"My MLP's Accuracy: {accuracy}")





# # # Print the shapes of each dataset to understand the dimensions
# # print("x_train shape:", img_train.shape)
# # print("y_train shape:", label_train.shape)
# # print("x_test shape:", img_test.shape)
# # print("y_test shape:", label_test.shape)

# # # Print the first image's pixel values and label for both training and test sets
# # print("First image in x_train (flattened):", img_train[0])
# # print("Label of first image in y_train:", label_train[0])

# # print("First image in x_test (flattened):", img_test[0])
# # print("Label of first image in y_test:", label_test[0])

# # # Optional: print a small subset of labels to get a sense of the data
# # print("First 10 labels in y_rain:", label_train[:10])
# # print("First 10 labels in y_test:", label_test[:10])




# # normalize = 1, use_rotation = 1, use validation set = 1
# preprocess = [1, 1, 0]

# img_train, label_train, img_test, label_test, img_val, label_val = loader.load_cifar10_data(cifar_dir, preprocess)


# img_train = img_train.reshape(-1, 32, 32, 3)
# img_test = img_test.reshape(-1, 32, 32, 3)

# CNN = NeuralNetworkCNN(dropout_rate=0.2)

# # Train the model
# CNN.train(img_train, label_train, epochs=50)

# # Evaluate the model
# accuracy, predictions = CNN.evaluate(img_test, label_test)
# print(f"My CNN's Accuracy: {accuracy}")