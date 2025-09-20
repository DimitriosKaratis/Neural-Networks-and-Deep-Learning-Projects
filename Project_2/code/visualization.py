import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from ola_edo.paths_config import Config


class Visualization:

    # Function that generates the confusion matrix
    def generate_conf_matrix(label_test, label_pred, name="conf_matrix.png", title=""):
        cm = confusion_matrix(label_test, label_pred, labels=np.unique(label_test))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(label_test))
        disp.plot(cmap=plt.cm.Blues)
        plt.title(title)

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

