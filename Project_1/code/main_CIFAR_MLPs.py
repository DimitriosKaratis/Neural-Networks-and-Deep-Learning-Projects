from dataset_functions import DatasetLoader
from visualization import Visualization
from myMLP import myMLP
from MLP import kerasMLP
from paths_config import Config
import numpy as np

  
# Load dataset
loader = DatasetLoader()
cifar_dir = Config.DATASET_PATH 

# preprocess = [norm, rot, valid, PCA]
num_of_classes = 10
preprocess = [1, 1, 1, 0]
shuffle_training_data = True

img_train, label_train, img_test, label_test, img_val, label_val, n_components, _, pca_object, scaler_object = loader.load_cifar10_data(cifar_dir, preprocess)

if n_components == 0:
    input_shape = 3072
else:
    input_shape = n_components


# MY MLP
my_mlp = myMLP(num_of_classes, [512,256,128], ['relu', 'relu', 'relu'], input_shape=input_shape, dropout_rate=0.3)
history_my_mlp = my_mlp.train(img_train, label_train, img_val, label_val, batch_size=32, epochs=40, shuffle_training=shuffle_training_data, early_stopping_patience=5)
accuracy_my_mlp, _, _ = my_mlp.evaluate(img_test, label_test)


# MLP using keras
mlp = kerasMLP(num_of_classes, [512,256,128], ['relu', 'relu', 'relu'], input_shape=(input_shape,), dropout_rate=0.3, l2_lambda=0.0)
history_mlp = mlp.train(img_train, label_train, img_val, label_val, batch_size=32, epochs=40, patience=10)
accuracy_mlp, predictions_mlp, _= mlp.evaluate(img_test, label_test)


# Test the network using random images from the web

# Define CIFAR-10 class names
cifar10_classes = [
    "airplane", "automobile", "bird", "cat", 
    "deer", "dog", "frog", "horse", "ship", "truck"
]

# Define custom test set
image_paths = Config.image_paths_mnist

custom_img_test = Visualization.load_custom_images(image_paths, flatten=True, preprocess=preprocess, pca=pca_object, scaler=scaler_object)
#custom_img_test = custom_img_test.reshape(custom_img_test.shape[0], -1)

# print(custom_img_test.shape)
# Define labels for the images using numeric labels
custom_label_test = np.array([5, 7, 7, 6, 2])  


# For MY MLP
accuracy_custom_my_mlp, predictions_classes_custom_my_mlp, _= my_mlp.evaluate(custom_img_test, custom_label_test)

# Map numeric labels to class names
predicted_labels_classes_my_mlp = [cifar10_classes[pred] for pred in predictions_classes_custom_my_mlp]
actual_labels_my_mlp = [cifar10_classes[label] for label in custom_label_test]

# Print results
# print(f"\nMy MLP's Accuracy: {accuracy_my_mlp}")
#print(f"My MLP's Accuracy on custom images: {accuracy_custom_my_mlp}")
#print("Predicted Labels:", predicted_labels_classes_my_mlp)
#print("Actual Labels:   ", actual_labels_my_mlp)



# For MLP using keras
accuracy_custom, predictions_classes_custom, _ = mlp.evaluate(custom_img_test, custom_label_test)

# Map numeric labels to class names
predicted_labels_classes = [cifar10_classes[pred] for pred in predictions_classes_custom]
actual_labels = [cifar10_classes[label] for label in custom_label_test]

# Print results

print(f"\nMy MLP's Accuracy: {accuracy_my_mlp}")
print(f"MLP's Accuracy using keras: {accuracy_mlp}")
#print(f"Keras MLP's Accuracy on custom images: {accuracy_custom}")
#print("Predicted Labels:", predicted_labels_classes)
#print("Actual Labels:   ", actual_labels)

my_mlp.plot_accuracy("my_mlp_cifar2.png")
mlp.plot_accuracy(history_mlp, "keras_mlp_cifar2.png")

Visualization.plot_comparison(history1=history_my_mlp, history2=history_mlp, model_name1="myMLP", model_name2="kerasMLP", name="comparison2.png")

