from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from paths_config import Config

class kerasCNN:
    def __init__(self, layers, activation_functions, dropout_rate=0.4, input_shape=(32, 32, 3)):
        self.model = Sequential()

        # Build the model dynamically
        for i, (filters, activation) in enumerate(zip(layers['conv'], activation_functions['conv'])):
            if i == 0:
                self.model.add(Conv2D(filters, (3, 3), activation=activation, padding='same', input_shape=input_shape))
            else:
                self.model.add(Conv2D(filters, (3, 3), activation=activation, padding='same'))
            self.model.add(BatchNormalization())
            self.model.add(Conv2D(filters, (3, 3), activation=activation, padding='same'))
            self.model.add(BatchNormalization())
            self.model.add(MaxPooling2D((2, 2)))
            self.model.add(Dropout(dropout_rate))
        
        # Fully Connected Layers
        self.model.add(Flatten())
        for neurons, activation in zip(layers['dense'], activation_functions['dense']):
            self.model.add(Dense(neurons, activation=activation))
            self.model.add(Dropout(dropout_rate))
        
        self.model.add(Dense(10, activation='softmax'))  # Output layer for 10 classes
        
        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, img_train, label_train, img_val, label_val, batch_size=32, epochs=50, patience=7):
        
        # Use Early Stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',  
            patience=patience,              
            # Restore model weights from the best epoch
            restore_best_weights=True, 
            verbose=1
        )

        # Train the model
        history = self.model.fit(
            img_train, label_train,
            validation_data=(img_val, label_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[early_stopping]  
        )

        return history

    # Evaluates the performance of the model
    def evaluate(self, img_test, label_test):
        label_pred_propability = self.model.predict(img_test)
        label_pred_classes = label_pred_propability.argmax(axis=1)  
        accuracy = accuracy_score(label_test, label_pred_classes)
        return accuracy, label_pred_classes, label_pred_propability

    # Predicts classes for the test dataset.
    def predict(self, img_test):
        return self.model.predict(img_test)

    # Plots training and validation accuracy
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
        plt.savefig(Config.RESULTS_PATH + name)  
        print("Plot saved as '" + name + "'")
