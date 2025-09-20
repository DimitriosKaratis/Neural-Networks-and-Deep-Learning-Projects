from ola_edo.dataset_functions import DatasetLoader
from ola_edo.SVM import SVMClassifier
from ola_edo.visualization import Visualization


# Load dataset
loader = DatasetLoader()

# preprocess = [use_normalization, use_rotation, use_PCA]
preprocess = [1, 1, 0.95]  

# Load CIFAR-10 dataset
img_train, label_train, img_test, label_test, _ = loader.load_cifar(preprocess)

# Verify the CIFAR-10 labels
#Visualization.verify_cifar10_labels(img_test[:3], label_test[:3], "cifar10.png", shape=(32, 32))

# Filter the dataset for two classes (e.g., class 0 and class 1)
num_classes = 2
img_train, label_train = loader.filter_classes(img_train, label_train, num_classes)
img_test, label_test = loader.filter_classes(img_test, label_test, num_classes)

#Initialize SVM classifier
Kernel = "polynomial"
svm = SVMClassifier(kernel=Kernel, degree=5)

# Perform grid search
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
gamma_values = [0.001, 0.01, 0.1, 1, 10, 100]
svm.grid_search(label_train.tolist(), img_train.tolist(), label_test.tolist(), img_test.tolist(), C_values, gamma_values)

# Plot grid search results
name = "grid_search_pca_no_rot_deg_5"
if Kernel == "linear":
    svm.plot_linear_grid_search_results(C_values, name)
elif Kernel == "rbf":
    svm.plot_grid_search_results(C_values, gamma_values, name)
elif Kernel == "polynomial":
    svm.plot_grid_search_results(C_values, gamma_values, name)






