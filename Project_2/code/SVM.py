import matplotlib.pyplot as plt
from itertools import product
from libsvm.svmutil import svm_train, svm_predict
from ola_edo.paths_config import Config
from sklearn.metrics import f1_score
import time

class SVMClassifier:
    def __init__(self, kernel="rbf", degree=3, alpha=0.40, beta=0.60, gamma=10):
        self.kernel = kernel  
        self.degree = degree  
        self.alpha = alpha  # Weight for training accuracy
        self.beta = beta    # Weight for F1 score
        self.gamma = gamma  # Weight for number of support vectors penalty
        self.best_params = {}
        self.best_score = 0
        self.grid_results = {}
        self.final_model = None

        # Define CIFAR-10 class names
        self.cifar10_classes = [
            "airplane", "automobile", "bird", "cat", 
            "deer", "dog", "frog", "horse", "ship", "truck"
            ]

    # Performs grid search over the given hyperparameters.
    def grid_search(self, train_labels, train_features, test_labels, test_features, C_values, gamma_values):
        print("\nPerforming Grid Search...\n")
        for C, gamma in product(C_values, gamma_values):
            if self.kernel == "rbf":
                param_str = f"-s 0 -t 2 -c {C} -g {gamma} -q"
                print(f"Testing parameters: C={C}, gamma={gamma}, kernel=rbf")
            elif self.kernel == "linear":
                param_str = f"-s 0 -t 0 -c {C} -q"
                print(f"Testing parameters: C={C}, kernel=linear")
            elif self.kernel == "polynomial":
                param_str = f"-s 0 -t 1 -c {C} -g {gamma} -d {self.degree} -q"
                print(f"Testing parameters: C={C}, gamma={gamma}, degree={self.degree}, kernel=poly")
            else:
                continue
            

            # Record the start time for this grid search iteration
            start_time = time.time()            
            temp_model = svm_train(train_labels, train_features, param_str)
            # Record the end time and calculate the elapsed time
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed Time: {elapsed_time:.2f} seconds")

            self.final_model = temp_model

            accuracy_train = svm_predict(train_labels, train_features, temp_model) 
            num_support_vectors, accuracy, predicted_labels, _ = self.evaluate(test_labels, test_features)
            self.grid_results[(C, gamma)] = accuracy[0]

            f1 = f1_score(test_labels, predicted_labels, average='weighted')  
            print(f"F1 Score: {f1:.4f}")  
    
            # Penalize for more support vectors
            support_vector_penalty = self.gamma * (num_support_vectors / len(train_labels))

            # Calculate the combined score (test accuracy + F1 + support vector penalty)
            combined_score = self.alpha * accuracy[0] + self.beta * f1 - support_vector_penalty

            # Map numeric labels to class names
            predicted_labels_classes = [self.cifar10_classes[int(pred)] for pred in predicted_labels[:10]]
            actual_labels = [self.cifar10_classes[label] for label in test_labels[:10]]


            print(f"Number of Support Vectors: {num_support_vectors}")
            print(f"Training Set Size: {len(train_labels)}")
            print(f"Percentage of Support Vectors: {100 * num_support_vectors / len(train_labels):.2f}%")

            print("Predicted Labels:", predicted_labels_classes)
            print("Actual Labels:   ", actual_labels)
            print("\n")

            # Update best params based on the combined score
            if combined_score > self.best_score:
                self.best_score = combined_score
                self.best_params = {'C': C, 'gamma': gamma}
                if self.kernel == "polynomial":
                    self.best_params['degree'] = self.degree

        print("\nGrid Search Complete")

        # Conditionally display parameters based on kernel type
        if self.kernel == "linear":
            print(f"Best Parameters: {{'C': {self.best_params['C']}}}")
        else:
            print(f"Best Parameters: {self.best_params}")
    

    # Evaluates the SVM model on the test set
    def evaluate(self, test_labels, test_features):
        print("\nEvaluating model on the test set...")
        predicted_labels, accuracy, pred_values = svm_predict(test_labels, test_features, self.final_model)
        num_support_vectors = self.final_model.get_nr_sv()

        return num_support_vectors, accuracy, predicted_labels, pred_values


    # Plots the accuracy results from the grid search (used for polynomial and rbf kernels).
    def plot_grid_search_results(self, C_values, gamma_values, name_prefix="grid_search"):
        plt.figure(figsize=(8, 6))
        plt.title(f"Grid Search Accuracy for {self.kernel.upper()} Kernel", fontsize=16)
        plt.xlabel("Gamma", fontsize=14)
        plt.ylabel("C", fontsize=14)

        accuracy_matrix = []
        for C in C_values:
            row = [self.grid_results[(C, gamma)] for gamma in gamma_values]
            accuracy_matrix.append(row)
        
        plt.imshow(accuracy_matrix, interpolation='nearest', cmap='viridis', origin='lower', aspect='auto',
                   extent=[min(gamma_values), max(gamma_values), min(C_values), max(C_values)])
        plt.colorbar(label="Accuracy (%)")
        plt.xticks(gamma_values, [f"{g:.2e}" for g in gamma_values], fontsize=12)
        plt.yticks(C_values, [f"{c}" for c in C_values], fontsize=12)
        plt.savefig(Config.RESULTS_PATH + f"{name_prefix}_{self.kernel}.png")
        print(f"Plot saved as '{name_prefix}_{self.kernel}.png'")


    # Plots the accuracy results from the grid search (used for linear kernels).
    def plot_linear_grid_search_results(self, C_values, name_prefix="linear_grid_search"):
        plt.figure(figsize=(8, 6))
        plt.title(f"Grid Search Accuracy for {self.kernel.upper()} Kernel", fontsize=16)
        plt.xlabel("C", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)

        accuracy_values = []
        gamma = next(iter(self.grid_results.keys()))[1]  # Dynamically determine gamma
        for C in C_values:
            accuracy_values.append(self.grid_results.get((C, gamma), 0))  # Default to 0 if not found

        # Use a logarithmic scale for the x-axis for better spacing
        plt.semilogx(C_values, accuracy_values, marker='o', linestyle='-', color='b')

        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.savefig(Config.RESULTS_PATH + f"{name_prefix}_{self.kernel}.png")
        plt.show()
        print(f"Plot saved as '{name_prefix}_{self.kernel}.png'")


