from dataset_functions import DatasetLoader
from myMLP import myMLP
from visualization import Visualization
import numpy as np
from paths_config import Config

  
# Load dataset
loader = DatasetLoader()
cifar_dir = Config.DATASET_PATH 

# preprocess = [norm, rot, valid, PCA]
num_of_classes = 10
preprocess = [1, 1, 1, 0.9]
shuffle_training_data = True

img_train, label_train, img_test, label_test, img_val, label_val, n_components, _, pca_object, scaler_object = loader.load_cifar10_data(cifar_dir, preprocess)

if n_components == 0:
    input_shape = 3072
else:
    input_shape = n_components



mlp = myMLP(num_of_classes, [512,256,128], ['relu','relu','relu'], input_shape=input_shape, dropout_rate=0.3)


mlp.train(img_train, label_train, img_val, label_val, batch_size=32, epochs=100, early_stopping_patience=5, shuffle_training=shuffle_training_data)
accuracy, labels_pred_classes, _ = mlp.evaluate(img_test, label_test)



# Test the network using random images from the web

# Define CIFAR-10 class names
cifar10_classes = [
    "airplane", "automobile", "bird", "cat", 
    "deer", "dog", "frog", "horse", "ship", "truck"
]

# Define custom test set
image_paths = Config.image_paths_cifar10

custom_img_test = Visualization.load_custom_images(image_paths, flatten=True, preprocess=preprocess, pca=pca_object, scaler=scaler_object)

# Define labels for the images using numeric labels
custom_label_test = np.array([7, 1, 2, 7, 1, 8, 0, 5, 6, 4])   

# Evaluate the MLP
accuracy_custom, predictions_classes_custom, predictions_propabilities_custom = mlp.evaluate(custom_img_test, custom_label_test)

# Map numeric labels to class names
predicted_labels_classes = [cifar10_classes[pred] for pred in predictions_classes_custom]
actual_labels = [cifar10_classes[label] for label in custom_label_test]

# Print results
print(f"\nMy MLP's Accuracy: {accuracy}")
print(f"MLP's Accuracy on custom images: {accuracy_custom}")
print("Predicted Labels:", predicted_labels_classes)
print("Actual Labels:   ", actual_labels)
#print("Propabilities for each class:\n", predictions_propabilities_custom)

# Plot the training and validation results
mlp.plot_accuracy("myMLP15.png")
Visualization.generate_conf_matrix(label_test, labels_pred_classes, title="Confusion Matrix for My Custom MLP")
