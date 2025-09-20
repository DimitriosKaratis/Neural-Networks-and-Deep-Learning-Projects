import os
import cv2
import numpy as np
import random
import pickle

import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from sklearn.model_selection import train_test_split

class DatasetLoader:
    def load_images_from_folder(self, folder_path, image_size=(128, 128)):
        images, labels = [], []
        for label_folder in os.listdir(folder_path):
            label_folder_path = os.path.join(folder_path, label_folder)
            if os.path.isdir(label_folder_path):
                for img_name in os.listdir(label_folder_path):
                    img_path = os.path.join(label_folder_path, img_name)
                    if img_path.endswith(".jpg"):
                        img = cv2.imread(img_path)
                        img = cv2.resize(img, image_size)
                        images.append(img.flatten())
                        labels.append(label_folder)
        return np.array(images), np.array(labels)

    def augment_brightness(self, folder_path, image_size=(128, 128), factor_range=(0.2, 1.7)):
        augmented_images = []
        for label_folder in os.listdir(folder_path):
            label_folder_path = os.path.join(folder_path, label_folder)
            if os.path.isdir(label_folder_path):
                for img_name in os.listdir(label_folder_path):
                    img_path = os.path.join(label_folder_path, img_name)
                    if img_path.endswith(".jpg"):
                        img = cv2.imread(img_path)
                        factor = random.uniform(*factor_range)
                        augmented_img = cv2.convertScaleAbs(img, alpha=factor, beta=0)
                        augmented_images.append(cv2.resize(augmented_img, image_size).flatten())
        return np.array(augmented_images)

    def augment_rotation(self, folder_path, image_size=(128, 128), angle_range=(-30, 30)):
        augmented_images = []
        for label_folder in os.listdir(folder_path):
            label_folder_path = os.path.join(folder_path, label_folder)
            if os.path.isdir(label_folder_path):
                for img_name in os.listdir(label_folder_path):
                    img_path = os.path.join(label_folder_path, img_name)
                    if img_path.endswith(".jpg"):
                        img = cv2.imread(img_path)
                        img = cv2.resize(img, image_size)
                        angle = random.uniform(*angle_range)
                        center = (image_size[0] // 2, image_size[1] // 2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                        rotated_img = cv2.warpAffine(img, rotation_matrix, image_size)
                        augmented_images.append(rotated_img.flatten())
        return np.array(augmented_images)


    # Opens different batches and return a dictonary with image data and labels
    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict   

    # Function to load cifar-10 data
    # preprocess is matrxix indicating: 
    # [using normalize, using roatations, using validation set for EarlyStopping]
    def load_cifar10_data(self, data_dir, preprocess):
        x_train, y_train, x_test, y_test, x_val, y_val = [], [], [], [], [], []

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
            x_train, x_test = self.normalize_pixels(x_train, x_test)
            if len(x_val) > 0:  # Normalize x_val if it exists
                x_val = self.normalize_pixels(x_val)

        return x_train, y_train, x_test, y_test, x_val, y_val

    

    # Normalizes pixel values
    def normalize_pixels(self, x_train, x_test):
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        return x_train, x_test



    def custom_rotation(self, img_dataset, label_dataset):
        """
        Applies random rotation between -10 and 10 degrees to each image in the dataset 
        and returns the rotated dataset along with the corresponding labels.

        Parameters:
        - img_dataset: 2D numpy array where each row is a flattened image (e.g., shape (50000, 3072)).
        - label_dataset: 1D numpy array containing the labels corresponding to the images (e.g., shape (50000,)).

        Returns:
        - rotated_img_dataset: 2D numpy array with rotated images, same shape as input dataset.
        - rotated_label_dataset: 1D numpy array with the corresponding labels for the rotated images.
        """
        rotated_img_dataset = []
        rotated_label_dataset = []

        for flattened_image, label in zip(img_dataset, label_dataset):
            # Reshape the flattened image to (32, 32, 3)
            reshaped_image = flattened_image.reshape((32, 32, 3))
            
            # Generate a random angle between -10 and 10 degrees, excluding 0 degrees
            random_angle = np.random.uniform(-10, 10)
            while random_angle == 0:  # Avoid 0 degrees
                random_angle = np.random.uniform(-10, 10)
            
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



