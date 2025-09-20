from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from paths_config import Config

class MNISTAutoencoder:
    def __init__(self):
        self.autoencoder = self._build_model()

    def _build_model(self):
        # Define the autoencoder model
        input_img = layers.Input(shape=(28, 28, 1))

        # Encoder
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Flatten()(x)
        latent = layers.Dense(32, activation='relu')(x)

        # Decoder
        x = layers.Dense(7 * 7 * 16, activation='relu')(latent)
        x = layers.Reshape((7, 7, 16))(x)
        x = layers.Conv2DTranspose(16, (3, 3), activation='relu', strides=2, padding='same')(x)
        x = layers.Conv2DTranspose(8, (3, 3), activation='relu', strides=2, padding='same')(x)
        output_img = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        # Create the model
        model = models.Model(input_img, output_img)
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, x_train, y_train_images, x_test, y_test_images, epochs=10, batch_size=32):
        history = self.autoencoder.fit(
            x_train, y_train_images,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test_images)
        )
        return history
    
    def predict(self, x):
        return self.autoencoder.predict(x)
   
    def plot_loss(history, name):
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(Config.RESULTS_PATH + name)  
        print("Plot saved as '" + name + "'")



    def visualize_results(x_test, y_test, y_pred_images, name, num_samples=10):
        indices = np.arange(num_samples)

        plt.figure(figsize=(15, 6))
        for i, index in enumerate(indices):
            input_image = x_test[index].reshape(28, 28)
            generated_image = y_pred_images[index].reshape(28, 28)

            plt.subplot(2, num_samples, i + 1)
            plt.imshow(input_image, cmap='gray')
            plt.title(f'Input: {y_test[index]}')
            plt.axis('off')

            plt.subplot(2, num_samples, i + 1 + num_samples)
            plt.imshow(generated_image, cmap='gray')
            plt.title(f'Gen/d: {(y_test[index] + 1) % 10}')  
            plt.axis('off')

        plt.tight_layout()
      
        plt.savefig(Config.RESULTS_PATH + name)  
        print("Plot saved as '" + name + "'")

    