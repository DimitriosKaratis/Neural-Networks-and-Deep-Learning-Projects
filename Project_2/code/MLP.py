import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from ola_edo.paths_config import Config


class kerasMLP:
    def __init__(self, num_classes=10, neurons=512, learning_rate=0.01, dropout_rate=0.0, input_shape=(3072,)):
        # Define the MLP model
        self.model = Sequential([
            Dense(neurons, activation='relu', input_shape=input_shape),
            Dropout(dropout_rate),                                       
            Dense(num_classes, activation='linear')  # Output layer with linear activation (required for hinge loss)
        ])

        # Compile the model with hinge loss
        self.model.compile(
            optimizer=SGD(learning_rate=learning_rate, momentum=0.9),  # Use SGD optimizer
            loss='hinge',                                     # Hinge loss
            metrics=['accuracy']                              # Track accuracy for evaluation
        )



    def train(self, img_train, label_train, img_val, label_val, batch_size=32, epochs=100, patience=5):
        # Define the EarlyStopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',   # Monitor the validation loss
            patience=patience,           # Stop training if val_loss doesn't improve for 5 epochs
            restore_best_weights=True  # Restore model weights from the best epoch
        )

        # Train the model
        history = self.model.fit(
            img_train, label_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(img_val, label_val),
            callbacks=[early_stopping]  
        )
                       
        return history


    def predict(self, img_test):
        # Get predictions
        predictions = self.model.predict(img_test)
        label_pred = np.argmax(predictions, axis=1)
        return predictions, label_pred


    def evaluate(self, img_test, label_test):
        # Evaluate the model on the test set
        loss, accuracy = self.model.evaluate(img_test, label_test, verbose=0)
        return loss, accuracy


    def plot_accuracy(self, history, name="mlp.png", title=''):
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
        plt.title('Training and Validation Accuracy '+ title, fontsize=16)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()

        # Save the plot to a file
        plt.savefig(Config.RESULTS_PATH + name)  
        print("Plot saved as '" + name + "'")


   