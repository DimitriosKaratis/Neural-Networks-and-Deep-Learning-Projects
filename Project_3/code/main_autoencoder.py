import numpy as np
from tensorflow.keras.datasets import mnist
from autoencoder import MNISTAutoencoder
from sklearn.preprocessing import OneHotEncoder
from RBFNN import RBFNN

# Load the MNIST dataset
(img_train, label_train), (img_test, label_test) = mnist.load_data()

# Normalize and reshape the data for the autoencoder
img_train = img_train / 255.0
img_test = img_test / 255.0
img_train_reshaped = img_train.reshape(-1, 28, 28, 1)
img_test_reshaped = img_test.reshape(-1, 28, 28, 1)

# Prepare labels: Map each digit to its "next" digit
label_train_next = (label_train + 1) % 10
label_test_next = (label_test + 1) % 10

# Create a dictionary that maps each digit to its first occurrence in img_train_reshaped
digit_to_image = {}
for digit, image in zip(label_train.flatten(), img_train_reshaped):
    if digit not in digit_to_image:
        digit_to_image[digit] = image

# Prepare target images based on the "next" digit
label_train_images = np.array([digit_to_image[next_digit] for next_digit in label_train_next.flatten()])
label_test_images = np.array([digit_to_image[next_digit] for next_digit in label_test_next.flatten()])

# Initialize the autoencoder model
autoencoder = MNISTAutoencoder()

# Train the autoencoder
history = autoencoder.train(
    img_train_reshaped, label_train_images, img_test_reshaped, label_test_images, epochs=10, batch_size=32
)

# Plot training and validation loss
loss_plot_name = "autoencoder_loss.png"
MNISTAutoencoder.plot_loss(history, loss_plot_name)

# Test the autoencoder: Predict the next digit images for the test set
label_pred_images = autoencoder.predict(img_test_reshaped)


# For visualization purposess

# Denormalize the images for visualization
label_pred_images_vis = (label_pred_images * 255).astype(np.uint8)
# Visualize results
visualization_plot_name = "generate_next_digit_examples.png"
MNISTAutoencoder.visualize_results(img_test_reshaped, label_test, label_pred_images_vis, visualization_plot_name, num_samples=20)



# Flatten the predicted images for RBFNN testing
label_pred_flattened = label_pred_images.reshape(label_pred_images.shape[0], -1)  # Shape: (num_samples, 784)

# Prepare MNIST dataset for RBFNN training
img_train_flattened = img_train.reshape(img_train.shape[0], -1)  
img_test_flattened = img_test.reshape(img_test.shape[0], -1)     

# Reshape labels for one-hot encoding
label_train = label_train.reshape(-1, 1)
label_test = label_test.reshape(-1, 1)

# One-hot encode labels for RBFNN training
encoder = OneHotEncoder(sparse_output=False)
label_train_encoded = encoder.fit_transform(label_train)

# Define a list of number of centers to evaluate
num_centers_list = [1000]

# Train the RBFNN on original MNIST data
centr_method = 'kmeans'
rbfnn = RBFNN(center_method=centr_method, cifar=False)
rbfnn.train_and_evaluate_rbfnn(
    rbfnn, num_centers_list, img_train_flattened, label_train, label_train_encoded, img_test_flattened, label_test
)

# Test the RBFNN on autoencoder-predicted images
rbfnn.evaluate(rbfnn, num_centers_list[0], label_pred_flattened, label_test_next)
