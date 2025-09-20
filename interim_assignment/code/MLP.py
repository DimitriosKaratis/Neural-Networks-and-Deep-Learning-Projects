import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.metrics import accuracy_score
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping


class NeuralNetworkMLP:
    def __init__(self, num_of_classes=10, layer_sizes=[64], activation_functions=['relu'], dropout_rate=0.0, optimizer='adam', input_shape=(3072,)):
        """
        Initializes the neural network for classification.

        Parameters:
        - num_of_classes: Number of classes for classification (e.g., 10 for CIFAR-10).
        - layer_sizes: List of integers where each integer represents the number of neurons in that layer.
        - activation_functions: List of activation functions for each layer. The length should match layer_sizes.
        - dropout_rate: Dropout rate for the Dropout layer to prevent overfitting.
        - input_shape: Tuple representing the shape of the input data.
        """
        self.model = Sequential()

        if len(layer_sizes) != len(activation_functions):
            raise ValueError("The length of layer_sizes and activation_functions must be the same.")
        
        # Define input layer with input shape
        self.model.add(Dense(layer_sizes[0], activation=activation_functions[0], input_shape=input_shape))
        self.model.add(Dropout(dropout_rate))
        
        # Add the remaining layers
        for size, activation in zip(layer_sizes[1:], activation_functions[1:]):
            self.model.add(Dense(size, activation=activation))
            self.model.add(Dropout(dropout_rate))
        
        # Output layer with 'num_of_classes' neurons for classification output
        self.model.add(Dense(num_of_classes, activation='softmax'))
        
        # Compile the model with optimizer and categorical loss for classification
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    def train(self, img_train, label_train, img_val, label_val, batch_size=32, epochs=500, patience=10):
        """
        Train the model with early stopping on the validation set.

        Parameters:
        - img_train: Training images.
        - label_train: Training labels.
        - img_val: Validation images.
        - label_val: Validation labels.
        - batch_size: Batch size for training.
        - epochs: Maximum number of epochs.
        - patience: Number of epochs to wait for improvement before stopping.
        """
        # EarlyStopping callback to stop training when validation loss stops improving
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        # Train the model with early stopping
        self.model.fit(img_train, label_train, epochs=epochs, batch_size=batch_size, verbose=1,
                       validation_data=(img_val, label_val), callbacks=[early_stopping])
        return self.model

    def predict(self, img_test):
        return self.model.predict(img_test)

    def evaluate(self, img_test, label_test):
        label_pred = self.predict(img_test)
        label_pred_classes = np.argmax(label_pred, axis=1)
        accuracy = accuracy_score(label_test, label_pred_classes)
        return accuracy, label_pred

