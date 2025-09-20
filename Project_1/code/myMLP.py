import numpy as np
import matplotlib.pyplot as plt

from paths_config import Config

class myMLP:

    # Initializes my Multi-Layer Perceptron model.
    # Parameters:
    #   - num_classes: Number of output classes for classification.
    #   - layer_sizes: List containing the number of neurons for each hidden layer.
    #   - activations: List of activation functions for each layer (e.g., 'relu', 'tanh').
    #   - input_shape: Dimension of the input vector (e.g., 3072 for CIFAR-10 flattened images).
    #   - dropout_rate: Probability of dropping neurons during training to prevent overfitting.
    #   - learning_rate: Learning rate for gradient descent optimization.        
    def __init__(self, num_classes=10, layer_sizes=[64], activations=['relu'], input_shape=3072, dropout_rate=0.0, learning_rate=0.01):
        self.num_classes = num_classes
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate  
        self.activations = activations  
        self.weights, self.biases = self._initialize_parameters(input_shape)
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}  # Store training/validation history


    # Initializes weights and biases using Xavier initialization for all layers.
    def _initialize_parameters(self, input_shape):
        weights = []
        biases = []

        input_shape = int(input_shape)
        layer_sizes = [input_shape] + [int(size) for size in self.layer_sizes] + [self.num_classes]

        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            stddev = np.sqrt(2.0 / (fan_in + fan_out))
            weights.append(np.random.randn(fan_in, fan_out) * stddev)
            biases.append(np.zeros((1, fan_out)))  # Biases remain zero-initialized

        return weights, biases
    
    
    # Applies dropout by randomly zeroing out a fraction of activations to prevent overfitting.
    def _apply_dropout(self, activations, rate):
        mask = np.random.rand(*activations.shape) > rate
        # Scale activations during training
        scaled_activations = activations * mask / (1 - rate)
        return scaled_activations


    # Relu activation function
    def _relu(self, z):
        return np.maximum(0, z)
    

    # Relu activation function's derivative
    def _relu_derivative(self, z):
        return (z > 0).astype(float)
    

    # Tanh activation function
    def _tanh(self, z):
        return np.tanh(z)
    

    # Tanh activation function's derivative
    def _tanh_derivative(self, z):
        return 1 - np.tanh(z)**2


    # Computes the softmax function for the output layer,
    # which converts raw scores into probabilities.
    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


    # Performs a forward pass through the network. 
    # Parameters:
    #   - X: Input data (batch).
    #   - training: Whether or not the forward pass is performed when training (enables dropout if True).        
    # Returns:
    #   - activations: List of activations at each layer.
    def _forward(self, X, training=True):
        activations = [X]
        for i, (w, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            X = np.dot(X, w) + b
            
            # Apply the corresponding activation function for the layer
            if i < len(self.activations):
                activation_func = self.activations[i]
                if activation_func == 'relu':
                    X = self._relu(X)
                elif activation_func == 'tanh':
                    X = self._tanh(X)
                else:
                    raise ValueError(f"Unsupported activation function: {activation_func}")
            
            # Apply dropout only during training
            if training and self.dropout_rate > 0:  
                X = self._apply_dropout(X, self.dropout_rate)

            activations.append(X)

        # Output layer (no dropout, and softmax activation)
        output = self._softmax(np.dot(X, self.weights[-1]) + self.biases[-1])
        activations.append(output)
        return activations
    

    # Computes the sparse_categorical_crossentropy loss between true labels and predicted probabilities.
    def _compute_loss(self, y_true, y_pred):
        predicted_probs = y_pred[range(len(y_true)), y_true]
        log_probs = np.log(predicted_probs)
        loss = -np.mean(log_probs)
        return loss

    # Performs the backward pass (backpropagation) to compute gradients for weights and biases.        
    # Parameters:
    #   - y: True labels for the batch.
    #   - activations: List of activations from the forward pass.
    #   - batch_size: Size of the current batch.        
    # Returns:
    #   - grads_w: Gradients for weights.
    #   - grads_b: Gradients for biases.
    def _backward(self, y, activations, batch_size):
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]

        delta = activations[-1]
        delta[range(batch_size), y] -= 1
        grads_w[-1] = np.dot(activations[-2].T, delta) / batch_size
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True) / batch_size

        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            # Get the derivative of the activation function
            if self.activations[i] == 'relu':
                derivative = self._relu_derivative(activations[i + 1])
            elif self.activations[i] == 'tanh':
                derivative = self._tanh_derivative(activations[i + 1])
            else:
                raise ValueError(f"Unsupported activation function: {self.activations[i]}")
            
            delta = np.dot(delta, self.weights[i + 1].T) * derivative
            grads_w[i] = np.dot(activations[i].T, delta) / batch_size
            grads_b[i] = np.sum(delta, axis=0, keepdims=True) / batch_size

        return grads_w, grads_b
    

    # Updates model parameters (weights and biases) using gradient descent.
    def _update_parameters(self, grads_w, grads_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]


    # Trains the MLP using gradient descent on the given training data.
    # Parameters:
    #   - X_train: Training data inputs.
    #   - y_train: Training data labels.
    #   - X_val: Validation data inputs.
    #   - y_val: Validation data labels. 
    #   - batch_size: Size of each mini-batch for gradient descent.
    #   - epochs: Number of epochs to train the model.
    #   - early_stopping_patience: Number of epochs to wait before stopping when validation loss doesn't improve.
    #   - shuffle_training: Boolean variable that indicates if random shuffling on the dataset will be performed or not.
    # Returns: history
    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=100, early_stopping_patience=5, shuffle_training=False):
        num_samples = X_train.shape[0]
        best_val_loss = float('inf')
         # Keeps track of how many epochs we've gone without improvement
        patience_counter = 0 
        
        for epoch in range(epochs):

            if shuffle_training:
                #Shuffle training data
                np.random.seed(42)
                indices = np.random.permutation(num_samples)
                X_train = X_train[indices]
                y_train = y_train[indices]

            # Mini-batch training
            for i in range(0, num_samples, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                # We are in training mode (training=True)
                activations = self._forward(X_batch, training=True) 
                grads_w, grads_b = self._backward(y_batch, activations, batch_size)
                self._update_parameters(grads_w, grads_b)

            # Calculate training metrics, disable dropout (training=False)
            train_activations = self._forward(X_train, training=False)  
            train_loss = self._compute_loss(y_train, train_activations[-1])
            train_acc = np.mean(np.argmax(train_activations[-1], axis=1) == y_train)

            # Calculate validation metrics, disable dropout (training=False)
            val_activations = self._forward(X_val, training=False)  
            val_loss = self._compute_loss(y_val, val_activations[-1])
            val_acc = np.mean(np.argmax(val_activations[-1], axis=1) == y_val)

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Reset counter if validation loss improves
                patience_counter = 0  
            else:
                patience_counter += 1

            # Print metrics
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Check if early stopping is triggered
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        return self.history
    

    # Performs a forward pass through the network to generate predictions.
    # Parameters:
    #   - X: Input data (numpy array of shape [num_samples, input_features]).
    # Returns:
    #   - A 1D numpy array of predicted class indices for each input sample.
    #     Each index corresponds to the class with the highest predicted probability.
    def predict(self, X):
        activations = self._forward(X, training=False)
        predictions = activations[-1]
        return predictions


    # Evaluates the model's performance on the given dataset.
    # Parameters:
    #   - X_test: Input data to evaluate.
    #   - y_test: True labels for the input data.
    # Returns:
    #   - accuracy: Classification accuracy on the dataset.
    #   - y_pred_classes: The predictions of the model in terms of classes. For example ([1, 0, 3, ...]),
    #     meaning that the first image belongs to class 1 etc.
    #   - y_pred_propability: The predictions of the model in terms of propabilities. For example ([0.1, 0.3...], [0.02, ...], ...),
    #     meaning that for the first image there is a 0.1 propability for it to be in the first class, 0.3 to be in the second class etc. Used for debugging
    def evaluate(self, X_test, y_test):
        y_pred_propability = self.predict(X_test)
        y_pred_classes = np.argmax(y_pred_propability, axis=1)
        accuracy = np.mean(y_pred_classes == y_test)
        return accuracy, y_pred_classes, y_pred_propability     


    # Plots the training and validation loss and accuracy over epochs.
    def plot_accuracy(self, name="plot.png", title=''):
        epochs = range(1, len(self.history['train_acc']) + 1)

        plt.figure(figsize=(10, 6))

        # Plot training accuracy
        plt.plot(epochs, self.history['train_acc'], label='Training Accuracy', marker='o', linestyle='-', color='blue')

        # Plot validation accuracy
        plt.plot(epochs, self.history['val_acc'], label='Validation Accuracy', marker='o', linestyle='-', color='orange')

        # Title and labels
        plt.title('Training and Validation Accuracy '+ title, fontsize=16)
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)

        # Add grid and legend
        plt.grid(visible=True, linestyle='--', linewidth=0.5)
        plt.legend(fontsize=12)

        # Tight layout for better spacing
        plt.tight_layout()
       
        # Save the plot to a file
        plt.savefig(Config.RESULTS_PATH + name)  
        print("Plot saved as '" + name + "'")
