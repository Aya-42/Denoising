from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

# Load the saved model
model = load_model('denoising_autoencoder.h5')

# Load the MNIST dataset
(_, _), (x_test, _) = mnist.load_data()

# Normalize the pixel values between 0 and 1
x_test = x_test.astype('float32') / 255.

# Add noise to the images
noise_factor = 0.5
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Use the model to denoise the images
decoded_imgs = model.predict(x_test_noisy)

# Display some sample results
n = 10  # Number of images to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display denoised images
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
