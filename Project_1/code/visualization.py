import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image

from paths_config import Config


class Visualization:

    # Function that generates the confusion matrix
    def generate_conf_matrix(label_test, label_pred, name="conf_matrix.png", title=""):
        cm = confusion_matrix(label_test, label_pred, labels=np.unique(label_test))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(label_test))
        disp.plot(cmap=plt.cm.Blues)
        plt.title(title)

        plt.savefig(Config.RESULTS_PATH + name)  
        print("Plot saved as '" + name + "'")
    


    # Plots the training and validation accuracy for two models on the same plot.
    # Parameters:
    #   - history1: History dictionary for the first model containing 'train_acc' and 'val_acc'.
    #   - history2: History dictionary for the second model containing 'train_acc' and 'val_acc'.
    #   - model_name1: String, name of the first model.
    #   - model_name2: String, name of the second model.
    #   - name: String, name of the file to save the plot.
    #   - title: String, title of the plot.
    def plot_comparison(history1, history2, model_name1="Model 1", model_name2="Model 2", name="plot.png", title=""):
        plt.figure(figsize=(12, 6))

        # Plot Model 1 (Blue)
        plt.plot(history1['train_acc'], label=f'{model_name1} - Training Accuracy', color='blue', marker='o', linestyle='-')
        plt.plot(history1['val_acc'], label=f'{model_name1} - Validation Accuracy', color='blue', marker='o', linestyle='--')

        # Plot Model 2 (Orange)
        plt.plot(history2.history['accuracy'], label=f'{model_name2} - Training Accuracy', color='orange', marker='x', linestyle='-')
        plt.plot(history2.history['val_accuracy'], label=f'{model_name2} - Validation Accuracy', color='orange', marker='x', linestyle='--')

        # Add labels, title, and legend
        plt.title(title if title else 'Training and Validation Accuracy Comparison', fontsize=16)
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(visible=True, linestyle='--', linewidth=0.5)

        # Save and display the plot
        plt.savefig(Config.RESULTS_PATH + name)
        print("Plot saved as '" + name + "'")


    # preprocess = [norm, rot, valid, PCA]
    def load_custom_images(image_paths, flatten=False, shape=(32,32), preprocess=[1,1,1,0], pca=None, scaler=None):
        images = []
        for path in image_paths:
            # Open image
            img = Image.open(path) 
            # Resize
            img = img.resize(shape)  
            img = np.array(img)

            # Normalize pixel values (0 to 1) if needed
            if preprocess[0] == 1:
                img = img / 255.0  
            # Flatten the image to 1D if we are using MLP
            if flatten:
                img = img.flatten() 

            images.append(img)

        images = np.array(images)

        # Apply the saved PCA and Scaler transformations if needed
        if pca is not None:
            images = scaler.transform(images)
            images = pca.transform(images)

        return images
        
    
    # Loads custom images formatted for the MNIST dataset.
    # Parameters:
    #   - image_paths (list of str): Paths to the image files.
    #   - flatten (bool): Whether to flatten the images to 1D arrays.
    #   - shape (tuple): Desired shape for resizing (default is (28, 28)).
    # Returns:
    #    - np.ndarray: Array of processed images.
    def load_mnist_custom_images(image_paths, flatten=False, shape=(28, 28), invert_colors=True):
        images = []
        for path in image_paths:
            try:
                # Open the image file
                img = Image.open(path) 
                # Convert to grayscale
                img = img.convert('L')  
                # Resize to MNIST dimensions
                img = img.resize(shape)  
                # Convert to NumPy array
                img = np.array(img)

                # Invert colors if required
                if invert_colors:
                    img = Visualization.invert_colors(img)  

                # Normalize pixel values to range [0, 1]    
                img = img / 255.0  

                # Flatten the image if required
                if flatten:
                    img = img.flatten()  
                images.append(img)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
        return np.array(images)

    # Function that inverts the colors of an image
    def invert_colors(image_array):
        return 255 - image_array


    # Visualize images with their corresponding labels to ensure correctness.
    # Parameters:
    #   - images: Array of image data (flattened or not).
    #   - labels: Array of labels corresponding to the images.
    #   - shape: Shape of each image for reshaping (default is 28x28 for MNIST).
    def verify_labels_mnist(images, labels, name="mnist_verify_data.png", shape=(28, 28)):
        num_images = len(images)
        # Adjust figure size based on the number of images
        plt.figure(figsize=(10, 2 * (num_images // 5 + 1)))  
        
        for i in range(num_images):
            plt.subplot(1, num_images, i + 1)
            # If flattened, reshape to original dimensions
            if len(images[i].shape) == 1: 
                plt.imshow(images[i].reshape(shape), cmap='gray')
            # If already in shape
            else: 
                plt.imshow(images[i], cmap='gray')
            plt.title(f"Label: {labels[i]}")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(Config.RESULTS_PATH + name)  
        print("Plot saved as '" + name + "'")


    # Visualize images with their corresponding labels to ensure correctness for CIFAR-10 dataset.
    # Parameters:
    #   - images: Array of image data (normalized or not).
    #   - labels: Array of labels corresponding to the images.
    #   - shape: Shape of each image for reshaping (default is 32x32 for CIFAR-10).
    def verify_cifar10_labels(images, labels, name="cifar10_preview.png", shape=(32, 32)):
        num_images = len(images)
        # Adjust figure size based on the number of images
        plt.figure(figsize=(10, 2 * (num_images // 5 + 1))) 

        for i in range(num_images):
            plt.subplot(1, num_images, i + 1)
            
            # If flattened, reshape to original dimensions
            img = images[i]
            if len(img.shape) == 1:  
                img = img.reshape(shape + (3,))  # CIFAR-10 images have 3 color channels

            # If normalized, multiply by 255 to convert back to [0, 255] range
            if img.max() <= 1:
                img = img * 255

            # Clip to avoid invalid pixel values
            img = np.clip(img, 0, 255).astype(np.uint8)

            plt.imshow(img)
            plt.title(f"Label: {labels[i]}")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(Config.RESULTS_PATH + name)  
        print(f"Plot saved as '{name}'")

