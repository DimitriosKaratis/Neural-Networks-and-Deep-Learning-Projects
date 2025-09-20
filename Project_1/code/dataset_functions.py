import numpy as np
import pickle
import matplotlib.pyplot as plt
import joblib

from scipy.ndimage import rotate
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from paths_config import Config


class DatasetLoader:

    # Opens different batches and return a dictonary with image data and labels
    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict   


    # Function to load cifar-10 data
    # preprocess is matrxix indicating: 
    # [using normalize, using roatations, using validation set for EarlyStopping, using PCA]
    def load_cifar10_data(self, data_dir, preprocess):
        x_train, y_train, x_test, y_test, x_val, y_val = [], [], [], [], [], []
        explained_variance_ratio = None
        pca_object = None
        scaler_object = None
        n_components = 0

        # Load training folder
        for i in range(1, 6):  # Load the 5 training batches
            batch = self.unpickle(f"{data_dir}/data_batch_{i}")
            x_train.append(batch[b'data'])
            y_train.extend(batch[b'labels'])
        x_train = np.concatenate(x_train)
        y_train = np.array(y_train)
        
        # Load test folder
        test_batch = self.unpickle(f"{data_dir}/test_batch")
        x_test = test_batch[b'data']
        y_test = np.array(test_batch[b'labels'])

        # Check if validation split is enabled
        if preprocess[2] == 1:
            # Split the training data into training and validation sets
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

        # Check if rotation augmentation is enabled
        if preprocess[1] == 1:
            x_train, y_train = self.custom_rotation(x_train, y_train)

        # Check if normalization is enabled
        if preprocess[0] == 1:
            x_train = self.normalize_pixels(x_train)
            x_test = self.normalize_pixels(x_test)
            # Normalize x_val if it exists
            if len(x_val) > 0:  
                x_val = self.normalize_pixels(x_val)
        
        # Check if PCA is enabled
        if preprocess[3] != 0.0:
            x_train, x_test, x_val, n_components, explained_variance_ratio, pca_object, scaler_object = self.apply_pca(x_train, x_test, x_val, preprocess[3])
            self.plot_cumulative_variance(explained_variance_ratio)

        return x_train, y_train, x_test, y_test, x_val, y_val, n_components, explained_variance_ratio, pca_object, scaler_object


    # Normalizes pixel values
    def normalize_pixels(self, x):
        x = x / 255.0
        return x
  

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
    def apply_pca(self, x_train, x_test, x_val, variance_threshold=0.95):
        # Flatten the images to 1D (3072 features per image)
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        x_test_flat = x_test.reshape(x_test.shape[0], -1)
        if len(x_val) > 0:  # Apply to validation data if it exists
            x_val_flat = x_val.reshape(x_val.shape[0], -1)
        else:
            x_val_flat = None

        # Standardize the data (mean=0, std=1)
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_flat)
        x_test_scaled = scaler.transform(x_test_flat)
        if x_val_flat is not None:
            x_val_scaled = scaler.transform(x_val_flat)
        else:
            x_val_scaled = None

        # Save the scaler
        joblib.dump(scaler, "scaler.pkl")    
        # Load the saved Scaler object
        scaler_object = joblib.load("scaler.pkl")

        # Apply PCA without specifying n_components to get all components
        pca = PCA()
        pca.fit(x_train_scaled)
        
        # Find the number of components to keep based on the variance threshold
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1  

        print(f"Optimal number of components: {n_components} for {variance_threshold*100}% variance")

        # Apply PCA
        pca = PCA(n_components=n_components)
        x_train_pca = pca.fit_transform(x_train_scaled)
        x_test_pca = pca.transform(x_test_scaled)
        if x_val_scaled is not None:
            x_val_pca = pca.transform(x_val_scaled)
        else:
            x_val_pca = None

        # Save the PCA object
        joblib.dump(pca, "pca_model.pkl")
        # Load the saved PCA object
        pca_object = joblib.load("pca_model.pkl")

        return x_train_pca, x_test_pca, x_val_pca, n_components, pca.explained_variance_ratio_, pca_object, scaler_object


    # Function to plot the cumulative variance after aplying PCA, just to visualize
    def plot_cumulative_variance(self, explained_variance_ratio, name="plot_pca.png"):
        # Assuming `explained_variance_ratio` is the result of PCA
        cumulative_variance = np.cumsum(explained_variance_ratio)

        # Plot the cumulative explained variance
        plt.plot(cumulative_variance)
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance by Principal Components')
        plt.grid(True)
       
        plt.savefig(Config.RESULTS_PATH + name)  
        print("Plot saved as '" + name + "'")