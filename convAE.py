# Convolutional Autoencoder over MNIST dataset
# Author: Shobhit Lamba
# e-mail: slamba4@uic.edu

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.datasets import mnist

input_img = Input(shape = (28, 28, 1)) 
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  

# Encoded mapping of the input
x = Conv2D(16, (3, 3), activation = "relu", padding = "same")(input_img)
x = MaxPooling2D((2, 2), padding = "same")(x)
x = Conv2D(8, (3, 3), activation = "relu", padding = "same")(x)
x = MaxPooling2D((2, 2), padding = "same")(x)
x = Conv2D(8, (3, 3), activation = "relu", padding = "same")(x)
encoded = MaxPooling2D((2, 2), padding = "same")(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional
# Lossy reconstruction of input
x = Conv2D(8, (3, 3), activation = "relu", padding = "same")(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation = "relu", padding = "same")(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation = "relu")(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation = "sigmoid", padding = "same")(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer = "adadelta", loss = "binary_crossentropy")

autoencoder.fit(x_train, x_train,
                epochs = 100,
                batch_size = 128,
                shuffle = True,
                validation_data = (x_test, x_test),
                callbacks = [TensorBoard(log_dir = "/tmp/autoencoder")])

decoded_imgs = autoencoder.predict(x_test)

# Displaying reconstruction using matplotlib
num_classes = 10
plt.figure(figsize = (20, 4))
for i in range(num_classes):
    # display original
    ax = plt.subplot(2, num_classes, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, num_classes, i + 1 + num_classes)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()