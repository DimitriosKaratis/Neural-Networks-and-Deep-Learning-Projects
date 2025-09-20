import tensorflow as tf
from ola_edo.dataset_functions import DatasetLoader
from ola_edo.MLP import kerasMLP
from sklearn.model_selection import train_test_split

# Load dataset
loader = DatasetLoader()

# preprocess = [use_normalization, use_rotation, use_PCA]
preprocess = [1, 0, 0.0]

# Load CIFAR-10 dataset
img_train, label_train, img_test, label_test, n_components = loader.load_cifar(preprocess)

if n_components == 0:
    input_shape = 3072
else:
    input_shape = n_components

# Filter the dataset for two classes (e.g., class 0 and class 1)
num_classes = 2
img_train, label_train = loader.filter_classes(img_train, label_train, num_classes)
img_test, label_test = loader.filter_classes(img_test, label_test, num_classes)

# Split the training data into training and validation sets
img_train, img_val, label_train, label_val = train_test_split(img_train, label_train, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding (required for hinge loss)
num_classes = 10
label_train_onehot = tf.keras.utils.to_categorical(label_train, num_classes)
label_test_onehot = tf.keras.utils.to_categorical(label_test, num_classes)
label_val_onehot = tf.keras.utils.to_categorical(label_val, num_classes)

# Initialize MLP model
MLP = kerasMLP(num_classes, neurons=512, dropout_rate=0.4, learning_rate=0.01, input_shape=(input_shape,))

history = MLP.train(img_train, label_train_onehot, img_val, label_val_onehot, batch_size=32, epochs=50, patience=5)

loss, accuracy = MLP.evaluate(img_test, label_test_onehot)

print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

MLP.plot_accuracy(history, "mlp_no_pca_no_rot_dp04.png")