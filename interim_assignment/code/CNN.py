import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import accuracy_score
import numpy as np

class NeuralNetworkCNN:
    def __init__(self, dropout_rate=0.0, input_shape=(32, 32, 3)):
        """
        Initializes the CNN for CIFAR-10 classification.

        Parameters:
        - dropout_rate: Dropout rate for the Dropout layers to prevent overfitting.
        - input_shape: Tuple representing the shape of the input data (e.g., CIFAR-10 has (32, 32, 3)).
        """
        self.model = Sequential()
        
        # First convolutional layer
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        self.model.add(MaxPooling2D((2, 2)))
        
        # Second convolutional layer
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        
        # Third convolutional layer
        self.model.add(Conv2D(64, (3, 3), activation='relu'))

        # Fourth convolutional layer
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        
        # Flatten the output from convolutional layers
        self.model.add(Flatten())
        
        # Fully connected (Dense) layer
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(dropout_rate))
        
        # Output layer with 10 neurons for CIFAR-10 (10 classes)
        self.model.add(Dense(10, activation='softmax'))
        
        # Compile the model with an appropriate loss function and optimizer
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, img_train, label_train, batch_size=64, epochs=300):
        self.model.fit(img_train, label_train, epochs=epochs, batch_size=batch_size, verbose=1)
        return self.model

    def predict(self, img_test):
        return self.model.predict(img_test)

    def evaluate(self, img_test, label_test):
        label_pred = self.model.predict(img_test)
        label_pred_classes = label_pred.argmax(axis=1)  # Get the predicted class for each image
        accuracy = accuracy_score(label_test, label_pred_classes)
        return accuracy, label_pred
