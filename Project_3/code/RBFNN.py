import numpy as np
from sklearn.cluster import KMeans
from paths_config import Config
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time

class RBFNN:
    def __init__(self, center_method='kmeans', cifar=True):
        self.center_method = center_method
        self.centers = None
        self.sigmas = None
        self.weights = None
        self.cifar = cifar

        # Define CIFAR-10 class names
        self.cifar10_classes = [
            "airplane", "automobile", "bird", "cat", 
            "deer", "dog", "frog", "horse", "ship", "truck"
            ]
        # Define MNIST class names
        self.mnist_classes = [str(i) for i in range(10)]

    def rbf_kernel(self, X, center, sigma):
        return np.exp(-np.linalg.norm(X - center, axis=1) ** 2 / (2 * sigma ** 2))

    def compute_interpolation_matrix(self, X, num_centers):
        G = np.zeros((X.shape[0], num_centers))
        for i, center in enumerate(self.centers):
            G[:, i] = self.rbf_kernel(X, center, self.sigmas[i])
        return G

    def fit(self, X, y, num_centers):
        if self.center_method == 'kmeans':
            # Use KMeans to determine centers
            kmeans = KMeans(n_clusters=num_centers, random_state=42).fit(X)
            self.centers = kmeans.cluster_centers_
        elif self.center_method == 'random':
            # Randomly select data points as centers
            random_indices = np.random.choice(X.shape[0], num_centers, replace=False)
            self.centers = X[random_indices]
        else:
            raise ValueError("Invalid center_method. Use 'kmeans' or 'random'.")

        # Compute the sigmas (widths of the Gaussian functions)
        sigmas = []
        for i in range(num_centers):
            distances = np.linalg.norm(self.centers - self.centers[i], axis=1)
            sigmas.append(np.mean(distances[distances > 0]))  # Exclude distance to itself
        self.sigmas = np.array(sigmas)

        # Compute the interpolation matrix G
        G = self.compute_interpolation_matrix(X, num_centers)

        # Compute the weights using pseudo-inverse
        self.weights = np.linalg.pinv(G).dot(y)


    def predict(self, X, num_centers):
        G = self.compute_interpolation_matrix(X, num_centers)
        return G.dot(self.weights)

    def train_and_evaluate_rbfnn(self, rbfnn, num_centers_list, x_train, y_train, y_train_encoded, x_test, y_test):
        test_accuracies = []
        train_accuracies = []

        # Ensure y_test is one-hot encoded before accessing it
        for num_centers in num_centers_list:
            
            # Record the start time for this grid search iteration
            start_time = time.time() 
            # Train the RBFNN model
            rbfnn.fit(x_train, y_train_encoded, num_centers)
            # Record the end time and calculate the elapsed time
            end_time = time.time()
            elapsed_time = end_time - start_time

            # Predictions on the training set
            y_train_pred = rbfnn.predict(x_train, num_centers)
            y_train_pred_labels = np.argmax(y_train_pred, axis=1)

            # Predictions on the test set
            y_test_pred = rbfnn.predict(x_test, num_centers)
            y_test_pred_labels = np.argmax(y_test_pred, axis=1)

            # Evaluate accuracy
            train_accuracy = accuracy_score(y_train.flatten(), y_train_pred_labels) 
            test_accuracy = accuracy_score(y_test.flatten(), y_test_pred_labels)  

            # Store the accuracies
            test_accuracies.append(test_accuracy)
            train_accuracies.append(train_accuracy)

            # Map predicted labels to class names
            if (self.cifar):
                predicted_labels_classes = [self.cifar10_classes[int(pred)] for pred in y_test_pred_labels[:10]]        
                actual_labels = [self.cifar10_classes[int(label)] for label in y_test[:10]]
            else:
                predicted_labels_classes = [self.mnist_classes[int(pred)] for pred in y_test_pred_labels[:10]]
                actual_labels = [self.mnist_classes[int(label)] for label in y_test[:10]]

            
            print("\nFor centr_method =", self.center_method, "and num_centers =", num_centers, ":")
            print(f"Elapsed Time: {elapsed_time:.2f} seconds")
            print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
            print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

            print("Predicted Labels:", predicted_labels_classes)
            print("Actual Labels:   ", actual_labels)

        return test_accuracies, train_accuracies

    def evaluate(self, rbfnn, num_centers, x_test, y_test):
        # Predictions on the test set
        y_test_pred = rbfnn.predict(x_test, num_centers)
        y_test_pred_labels = np.argmax(y_test_pred, axis=1)

        # Evaluate accuracy
        test_accuracy = accuracy_score(y_test.flatten(), y_test_pred_labels)

        # Map predicted labels to class names
        if (self.cifar):
                predicted_labels_classes = [self.cifar10_classes[int(pred)] for pred in y_test_pred_labels[:10]]        
                actual_labels = [self.cifar10_classes[int(label)] for label in y_test[:10]]
        else:
            predicted_labels_classes = [self.mnist_classes[int(pred)] for pred in y_test_pred_labels[:10]]
            actual_labels = [self.mnist_classes[int(label)] for label in y_test[:10]]


    
        print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
        print("Predicted Labels:", predicted_labels_classes)
        print("Actual Labels:   ", actual_labels)
    
    def plot_num_centers_effect(self, num_centers_list, test_accuracies, train_accuracies, name):
        # Plot the relationship between number of centers and accuracies
        plt.figure(figsize=(8, 6))
        plt.plot(num_centers_list, test_accuracies, marker='o', label='Test Accuracy', color='blue')
        plt.plot(num_centers_list, train_accuracies, marker='s', label='Training Accuracy', color='green')
        
        plt.xlabel('Number of Centers')
        plt.ylabel('Accuracy')
        plt.title('Effect of Number of Centers on Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(Config.RESULTS_PATH + self.center_method + "_" + name)  
        print("Plot saved as '" + self.center_method + "_" + name + "'")