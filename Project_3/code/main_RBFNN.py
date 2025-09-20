from sklearn.preprocessing import OneHotEncoder
from RBFNN import RBFNN
from dataset_functions import DatasetLoader


# Load dataset
loader = DatasetLoader()

# preprocess = [use_normalization, use_rotation, use_PCA]
preprocess = [1, 1, 0.95]  

# Load CIFAR-10 dataset
img_train, label_train, img_test, label_test, _ = loader.load_cifar(preprocess)

# Reshape label_train and label_test to be 2D arrays. Suitable for one-hot encoding
label_train = label_train.reshape(-1, 1)
label_test = label_test.reshape(-1, 1)

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
label_train_encoded = encoder.fit_transform(label_train)


# Define a list of number of centers to evaluate
num_centers_list = [10, 20, 50, 70, 100, 150, 200, 300, 500, 1000]

# Train and evaluate the RBFNN model
centr_method = 'random'
rbfnn = RBFNN(center_method=centr_method)
test_accuracies, train_accuracies = rbfnn.train_and_evaluate_rbfnn(rbfnn, num_centers_list, img_train, label_train, label_train_encoded, img_test, label_test)


# Plot number of centers vs. test accuracy
rbfnn.plot_num_centers_effect(num_centers_list, test_accuracies, train_accuracies, loader.name)