from dataset_functions import DatasetLoader
from visualization import Visualization
from myMLP import myMLP
from MLP import kerasMLP
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from paths_config import Config
  
# Load dataset
loader = DatasetLoader() 


num_of_classes = 10
shuffle_training_data = True
invert_colors = False

# MNIST TESTING
(img_train, label_train), (img_test, label_test) = tf.keras.datasets.mnist.load_data()
img_train = loader.normalize_pixels(img_train)
img_test = loader.normalize_pixels(img_test)
img_train = img_train.reshape(img_train.shape[0], -1)
img_test = img_test.reshape(img_test.shape[0], -1)
img_train, img_val, label_train, label_val = train_test_split(img_train, label_train, test_size=0.2, random_state=42)

# MY MLP
my_mlp = myMLP(num_of_classes, [512,256,128], ['relu', 'relu', 'relu'], input_shape=784, dropout_rate=0.3)
history_my_mlp = my_mlp.train(img_train, label_train, img_val, label_val, batch_size=32, epochs=50, shuffle_training=shuffle_training_data)
accuracy_my_mlp, _, _ = my_mlp.evaluate(img_test, label_test)


# MLP using keras
mlp = kerasMLP(num_of_classes, [512,256,128], ['relu', 'relu', 'relu'], input_shape=(784,), dropout_rate=0.3)
history_mlp = mlp.train(img_train, label_train, img_val, label_val, batch_size=32, epochs=50)
accuracy_mlp, _, _= mlp.evaluate(img_test, label_test)



# Test the network using random images from the web

# Define CIFAR-10 class names
mnist_classes = [
    "zero", "one", "two", "three", 
    "four", "five", "six", "seven", "eight", "nine"
]

# Define custom test set
image_paths = Config.image_paths_mnist


custom_img_test = Visualization.load_mnist_custom_images(image_paths, flatten=True, shape=(28,28), invert_colors=invert_colors)

# Define labels for the images using numeric labels
custom_label_test = np.array([5, 7, 7, 6, 2])  


# For MY MLP
accuracy_custom_my_mlp, predictions_classes_custom_my_mlp, _= my_mlp.evaluate(custom_img_test, custom_label_test)

# Map numeric labels to class names
predicted_labels_classes_my_mlp = [mnist_classes[pred] for pred in predictions_classes_custom_my_mlp]
actual_labels_my_mlp = [mnist_classes[label] for label in custom_label_test]

# Print results
print(f"\nMy MLP's Accuracy: {accuracy_my_mlp}")
print(f"My MLP's Accuracy on custom images: {accuracy_custom_my_mlp}")
print("Predicted Labels:", predicted_labels_classes_my_mlp)
print("Actual Labels:   ", actual_labels_my_mlp)


# Example usage
Visualization.verify_labels_mnist(img_test[:4], label_test[:4], "mnist_imgs.png", shape=(28, 28))
Visualization.verify_labels_mnist(custom_img_test, custom_label_test, "my_mnist_imgs.png", shape=(28, 28))


# For MLP using keras
accuracy_custom, predictions_classes_custom, _ = mlp.evaluate(custom_img_test, custom_label_test)

# Map numeric labels to class names
predicted_labels_classes = [mnist_classes[pred] for pred in predictions_classes_custom]
actual_labels = [mnist_classes[label] for label in custom_label_test]

# Print results
print(f"\nMLP's Accuracy using keras: {accuracy_mlp}")
print(f"Keras MLP's Accuracy on custom images: {accuracy_custom}")
print("Predicted Labels:", predicted_labels_classes)
print("Actual Labels:   ", actual_labels)

my_mlp.plot_accuracy("my_mlp_mnist.png")
mlp.plot_accuracy(history_mlp, "keras_mlp_mnist.png")
Visualization.plot_comparison(history1=history_my_mlp, history2=history_mlp, model_name1="myMLP", model_name2="kerasMLP", name="mnist_mlp_comparison.png")

