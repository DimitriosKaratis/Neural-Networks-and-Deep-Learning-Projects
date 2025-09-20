import numpy as np

from scipy.ndimage import rotate
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import MinMaxScaler



class DatasetLoader:

    # Function to load cifar-10 data
    # preprocess is matrxix indicating: 
    # [using normalize, using roatations, using PCA]
    def load_cifar(self, preprocess):
        n_components = 0

        # Load the CIFAR-10 dataset
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        # Flatten the image data for processing
        # Each image is reshaped from (32, 32, 3) to a single vector
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
    
        # Check if normalization is enabled 
        # If enabled, scale pixel values to the range [0, 1] using MinMaxScaler
        if preprocess[0] == 1:
            scaler = MinMaxScaler()
            x_train = scaler.fit_transform(x_train) 
            x_test = scaler.transform(x_test)       

        # Flatten label arrays to make them compatible with training functions
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        
        # Check if rotation augmentation is enabled 
        # If enabled, apply a custom rotation function to the training data
        if preprocess[1] == 1:
            x_train, y_train = self.custom_rotation(x_train, y_train)

        # Check if PCA is enabled 
        # If enabled, apply PCA to reduce dimensionality
        if preprocess[2] != 0.0:
            x_train, x_test, n_components = self.apply_pca(x_train, x_test, preprocess[2])
        
        # Return the preprocessed training and test datasets
        return x_train, y_train, x_test, y_test , n_components


    # Filters the dataset to include a specified number of classes.
    def filter_classes(self, img_data, label_data, num_classes=2):
        if num_classes < 2 or num_classes > 10:
            raise ValueError("num_classes must be between 2 and 10, inclusive.")

        # Select the first `num_classes` classes
        selected_classes = sorted(set(label_data.flatten()))[:num_classes]

        # Create a mask for the selected classes
        filter_mask = np.isin(label_data, selected_classes)

        # Filter the image and label data
        img_filtered = img_data[filter_mask.flatten()]
        label_filtered = label_data[filter_mask]

        # Relabel the classes to be sequential (0 to num_classes-1)
        class_mapping = {old_label: new_label for new_label, old_label in enumerate(selected_classes)}
        label_filtered = np.vectorize(class_mapping.get)(label_filtered)

        return img_filtered, label_filtered


    # Applies random rotation between -20 and 20 degrees to each image in the dataset 
    # and returns the rotated dataset along with the corresponding labels.
    # Parameters:
    #   - img_dataset: 2D numpy array where each row is a flattened image (e.g., shape (50000, 3072)).
    #   - label_dataset: 1D numpy array containing the labels corresponding to the images (e.g., shape (50000,)).
    #     Returns:
    #   - concatenated_img_dataset: 2D numpy array with the normal and rotated images, same shape as input dataset.
    #   - concatenated_label_dataset: 1D numpy array with the normal and corresponding labels for the rotated images.
    def custom_rotation(self, img_dataset, label_dataset):
        rotated_img_dataset = []
        rotated_label_dataset = []

        for flattened_image, label in zip(img_dataset, label_dataset):
            # Reshape the flattened image to (32, 32, 3)
            reshaped_image = flattened_image.reshape((32, 32, 3))
            
            # Generate a random angle between -20 and 20 degrees, excluding 0 degrees
            random_angle = np.random.uniform(-20, 20)
            while random_angle == 0:  # Avoid 0 degrees
                random_angle = np.random.uniform(-20, 20)
            
            # Rotate the image by the random angle (mode 'nearest' avoids padding with zeros)
            rotated_image = rotate(reshaped_image, random_angle, reshape=False, mode='nearest')
            
            # Flatten the rotated image back to (3072,)
            rotated_flattened_image = rotated_image.flatten()
            
            # Add the rotated image and its corresponding label to their respective lists
            rotated_img_dataset.append(rotated_flattened_image)
            rotated_label_dataset.append(label)

        # Convert lists to numpy arrays for proper concatenation
        img_dataset = np.array(img_dataset)
        label_dataset = np.array(label_dataset)
        rotated_img_dataset = np.array(rotated_img_dataset)
        rotated_label_dataset = np.array(rotated_label_dataset)

        # Concatenate the original dataset with the rotated one
        concatenated_img_dataset = np.concatenate((img_dataset, rotated_img_dataset), axis=0)
        concatenated_label_dataset = np.concatenate((label_dataset, rotated_label_dataset), axis=0)

        return concatenated_img_dataset, concatenated_label_dataset

    # Function to apply PCA on the data
    def apply_pca(self, x_train, x_test, variance_ratio):
        pca = PCA(n_components=variance_ratio)  # Retain the specified variance ratio
        x_train_pca = pca.fit_transform(x_train)
        x_test_pca = pca.transform(x_test)  # Apply the same PCA transformation to the test data
        
        n_components = pca.n_components_  # Number of components selected to preserve the variance ratio
        
        print(f"Optimal number of components: {n_components} for {variance_ratio*100}% variance")
        
        return x_train_pca, x_test_pca, n_components
