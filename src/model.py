import keras

"""
This model is a convolutional neural network to classify different pets. Each pet is an individual pet, not a breed.
There are 6 classes: dontcare, batima, botinha, linco, nico, peto.
The images are JPEGs with 3 channels (RGB) 800x600, the model is trained on 40x30.
"""

class Model(keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(40, 30, 3)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(6)
        ])

    def call(self, x):
        return self.model(x)
    
    def predict(self, x):
        return self.call(x)
    
    def plot(self, filename='model.png'):
        keras.utils.plot_model(self, to_file=filename, show_shapes=True, show_layer_names=True)

# Path: src/train_model.py
