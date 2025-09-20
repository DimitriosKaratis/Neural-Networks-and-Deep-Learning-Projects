from dataset_functions import DatasetLoader
from project.CNN import kerasCNN
import numpy as np
from visualization import Visualization
from paths_config import Config
  
# Load dataset
loader = DatasetLoader()
cifar_dir = Config.DATASET_PATH

# preprocess = [use_normalization, use_rotation, use_validation, use_PCA]
preprocess = [1, 1, 1, 0]

img_train, label_train, img_test, label_test, img_val, label_val, n_components, _, pca_object, scaler_object = loader.load_cifar10_data(cifar_dir, preprocess)


# Reshape data images
img_train = img_train.reshape(-1, 32, 32, 3)
img_test = img_test.reshape(-1, 32, 32, 3)
img_val = img_val.reshape(-1, 32, 32, 3)

# Define the CNN's parameters
layers = {
    'conv': [32, 64, 128],  # Number of filters for Conv2D layers
    'dense': [256]          # Number of neurons for Dense layers
}

activation_functions = {
    'conv': ['relu', 'relu', 'relu'],  # Activation functions for Conv2D layers
    'dense': ['relu']                  # Activation function for Dense layers
}

# Initialize the CNN object
cnn = kerasCNN(layers=layers, activation_functions=activation_functions, dropout_rate=0.3, input_shape=(32, 32, 3))

# Train the model
history = cnn.train(img_train, label_train, img_val, label_val, batch_size=32, epochs=40)

accuracy, labels_pred_classes, _ = cnn.evaluate(img_test, label_test)

# Test the network using random images from the web

# Define CIFAR-10 class names
cifar10_classes = [
    "airplane", "automobile", "bird", "cat", 
    "deer", "dog", "frog", "horse", "ship", "truck"
]

# Define custom test set
image_paths = Config.image_paths_cifar10

custom_img_test = Visualization.load_custom_images(image_paths, flatten=False, preprocess=preprocess, pca=pca_object, scaler=scaler_object)

# Define labels for the images using numeric labels
custom_label_test = np.array([7, 1, 2, 7, 1, 8, 0, 5, 6, 4])    

# Evaluate the MLP
accuracy_custom, predictions_classes_custom, predictions_propabilities_custom = cnn.evaluate(custom_img_test, custom_label_test)

# Map numeric labels to class names
predicted_labels_classes = [cifar10_classes[pred] for pred in predictions_classes_custom]
actual_labels = [cifar10_classes[label] for label in custom_label_test]

# Print results
print(f"\nCNN's Accuracy using keras: {accuracy}")
print(f"CNN's Accuracy on custom images: {accuracy_custom}")
print("Predicted Labels:", predicted_labels_classes)
print("Actual Labels:   ", actual_labels)
#print("Propabilities for each class:\n", predictions_propabilities_custom)

# Plot the training and validation results
cnn.plot_accuracy(history, "cnn1.png")
Visualization.generate_conf_matrix(label_test, labels_pred_classes, title="Confusion Matrix for CNN using keras")