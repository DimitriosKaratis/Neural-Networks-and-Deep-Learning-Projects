import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score

from paths_config import Config


class kerasMLP:
    def __init__(self, num_of_classes=10, layer_sizes=[64], activation_functions=['relu'], dropout_rate=0.0, optimizer='adam', input_shape=(3072,), l2_lambda=0.02):
        self.model = Sequential()

        if len(layer_sizes) != len(activation_functions):
            raise ValueError("The length of layer_sizes and activation_functions must be the same.")
        
        # Define input layer with input shape
        self.model.add(Dense(layer_sizes[0], activation=activation_functions[0], input_shape=input_shape, kernel_regularizer=l2(l2_lambda)))
        #self.model.add(Dropout(dropout_rate))
        
        # Add the remaining layers
        for size, activation in zip(layer_sizes[1:], activation_functions[1:]):
            self.model.add(Dense(size, activation=activation, kernel_regularizer=l2(l2_lambda)))
            
        self.model.add(Dropout(dropout_rate))
        
        # Output layer with 'num_of_classes' neurons for classification output
        self.model.add(Dense(num_of_classes, activation='softmax', kernel_regularizer=l2(l2_lambda)))
        
        # Compile the model with optimizer and categorical loss for classification
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    def train(self, img_train, label_train, img_val, label_val, batch_size=32, epochs=500, patience=5):
        # EarlyStopping callback to stop training when validation loss stops improving
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        # Train the model with early stopping
        history = self.model.fit(img_train, label_train, epochs=epochs, batch_size=batch_size, verbose=1,
                       validation_data=(img_val, label_val), callbacks=[early_stopping])
        return history

    def predict(self, img_test):
        return self.model.predict(img_test)

    def evaluate(self, img_test, label_test):
        label_pred_propability = self.predict(img_test)
        label_pred_classes = np.argmax(label_pred_propability, axis=1)
        accuracy = accuracy_score(label_test, label_pred_classes)
        return accuracy, label_pred_classes, label_pred_propability

    def plot_accuracy(self, history, name="plot.png", title=''):
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
        plt.title('Training and Validation Accuracy '+ title, fontsize=16)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()

        # Save the plot to a file
        plt.savefig(Config.RESULTS_PATH + name)  # Save to the current directory
        print("Plot saved as '" + name + "'")
