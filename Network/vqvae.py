import numpy as np
import tensorflow as tf
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Concatenate, Embedding, Add, Flatten
from tensorflow.keras.layers import Activation, LeakyReLU, Reshape
from tensorflow.keras.models import Model,load_model

from numpy.random import seed
seed(7)
tensorflow.random.set_seed(7)

latent_dim = 32
num_embeddings = 32
data_variance = 0.05

### encoder ###
encoder_inputs = Input(shape=(128, 128, 4),name="input",dtype="float32")
x = layers.Conv2D(64, 3, activation="leaky_relu", strides=2, padding="same")(encoder_inputs)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(128, 3, activation="leaky_relu", strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(256, 3, activation="leaky_relu", strides=3, padding="valid")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(512, 3, activation="leaky_relu", strides=2, padding='same')(x)
x = layers.BatchNormalization()(x)
encoder_outputs = layers.Conv2D(latent_dim, 1, activation="leaky_relu",strides=1, padding="same")(x)

encoder = Model(encoder_inputs,encoder_outputs, name="encoder")


### decoder ###

latent_inputs = Input(shape=encoder.output.shape[1:])

x = layers.Conv2DTranspose(512, 3, activation="leaky_relu", strides=1, padding="same")(latent_inputs)
x = layers.Conv2DTranspose(256, 3, activation="leaky_relu", strides=3, padding="valid",output_padding=1)(x)
x = layers.Conv2DTranspose(128, 3, activation="leaky_relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(64, 3, activation="leaky_relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="leaky_relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(4, 3, padding="same",name="output",activation='sigmoid')(x)#
decoder = Model(latent_inputs, decoder_outputs, name="decoder")


### functions

def get_vqvae(latent_dim=32, num_embeddings=32):
    vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
    encod = encoder
    decod = decoder
    inputs = Input(shape=(128, 128, 4))
    encoder_outputs = encod(inputs)
    quantized_latents = vq_layer(encoder_outputs)
    reconstructions = decod(quantized_latents)
    return Model(inputs, reconstructions, name="vq_vae")

def process(image):
    image = tf.cast(image/255. ,tf.float32)
    return image

def variance(image):
    image = tf.experimental.numpy.var(image)
    return image

def ResBlock(inputs,hidden):
    x = layers.Conv2D(hidden, 3, padding="same",strides=1, activation="relu")(inputs)
    x = layers.Conv2D(hidden, 3, padding="same",strides=1)(x)
    x = layers.Add()([inputs, x])
    return x

def Upsampling(inputs, hidden, factor=1):
    x = layers.Conv2D(hidden * (factor ** 2), 3, padding="same")(inputs)
    x = layers.Conv2D(hidden * (factor ** 2), 3, padding="same")(x)
    x = layers.Add()([inputs, x])
    return x


class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer.
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices


class VQVAETrainer(Model):
    def __init__(self, train_variance, vae, latent_dim=512, num_embeddings=64, **kwargs):
        super(VQVAETrainer, self).__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.vqvae = vae

        self.total_loss_tracker = tensorflow.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tensorflow.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = tensorflow.keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }

class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_epoch_begin(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        print("\nEpoch %05d: Learning rate is %6.8f." % (epoch, lr))