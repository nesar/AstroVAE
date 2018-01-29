"""
https://github.com/eriklindernoren/Keras-GAN/blob/master/aae/adversarial_autoencoder.py

Other implementations: https://github.com/bstriner/keras-adversarial
"""


from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K

import matplotlib.pyplot as plt
import numpy as np

import Cl_load


# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------


class AdversarialAutoencoder():
    def __init__(self):
        # self.img_rows = 28
        # self.img_cols = 28
        # self.channels = 1
        # self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.img_shape = (2549 ,1)
        self.encoded_dim = 10

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the encoder / decoder
        self.encoder = self.build_encoder()
        self.encoder.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

        self.decoder = self.build_decoder()
        self.decoder.compile(loss=['mse'],
            optimizer=optimizer)

        img = Input(shape=self.img_shape)
        # The generator takes the image, encodes it and reconstructs it
        # from the encoding
        encoded_repr = self.encoder(img)
        reconstructed_img = self.decoder(encoded_repr)

        # For the adversarial_autoencoder model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator determines validity of the encoding
        validity = self.discriminator(encoded_repr)

        # The adversarial_autoencoder model  (stacked generator and discriminator)
        self.adversarial_autoencoder = Model(img, [reconstructed_img, validity])
        self.adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[0.999, 0.001],
            optimizer=optimizer)


    def build_encoder(self):
        # Encoder
        encoder = Sequential()

        encoder.add(Flatten(input_shape=self.img_shape))
        encoder.add(Dense(512))
        encoder.add(LeakyReLU(alpha=0.2))
        encoder.add(BatchNormalization(momentum=0.8))
        encoder.add(Dense(512))
        encoder.add(LeakyReLU(alpha=0.2))
        encoder.add(BatchNormalization(momentum=0.8))
        encoder.add(Dense(self.encoded_dim))

        encoder.summary()

        img = Input(shape=self.img_shape)
        encoded_repr = encoder(img)

        return Model(img, encoded_repr)

    def build_decoder(self):
        # Decoder
        decoder = Sequential()

        decoder.add(Dense(512, input_dim=self.encoded_dim))
        decoder.add(LeakyReLU(alpha=0.2))
        decoder.add(BatchNormalization(momentum=0.8))
        decoder.add(Dense(512))
        decoder.add(LeakyReLU(alpha=0.2))
        decoder.add(BatchNormalization(momentum=0.8))
        decoder.add(Dense(np.prod(self.img_shape), activation='tanh'))
        decoder.add(Reshape(self.img_shape))

        decoder.summary()

        encoded_repr = Input(shape=(self.encoded_dim,))
        gen_img = decoder(encoded_repr)

        return Model(encoded_repr, gen_img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(512, input_dim=self.encoded_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1, activation="sigmoid"))
        model.summary()

        encoded_repr = Input(shape=(self.encoded_dim, ))
        validity = model(encoded_repr)

        return Model(encoded_repr, validity)

    def train(self, epochs, batch_size=128, save_interval=50):


        camb_in = Cl_load.cmb_profile(train_path = train_path,  train_target_path = train_target_path , test_path = test_path, test_target_path = test_target_path, num_para=5)

        (x_train, y_train), (x_test, y_test) = camb_in.load_data()

        x_train = x_train[:,2:]
        x_test = x_test[:,2:]

        print(x_train.shape, 'train sequences')
        print(x_test.shape, 'test sequences')
        print(y_train.shape, 'train sequences')
        print(y_test.shape, 'test sequences')

        normFactor = np.max( [np.max(x_train), np.max(x_test ) ])
        print('-------normalization factor:', normFactor)

        x_train = x_train.astype('float32')/normFactor #/ 255.
        x_test = x_test.astype('float32')/normFactor #/ 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


        


        # Load the dataset
        #(X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        #X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(x_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):


            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            # Generate a half batch of embedded images
            latent_fake = self.encoder.predict(imgs)

            latent_real = np.random.normal(size=(half_batch, self.encoded_dim))

            valid = np.ones((half_batch, 1))
            fake = np.zeros((half_batch, 1))

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(latent_real, valid)
            d_loss_fake = self.discriminator.train_on_batch(latent_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Generator wants the discriminator to label the generated representations as valid
            valid_y = np.ones((batch_size, 1))

            # Train the generator
            g_loss = self.adversarial_autoencoder.train_on_batch(imgs, [imgs, valid_y])

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                # Select a random half batch of images
                idx = np.random.randint(0, X_train.shape[0], 25)
                imgs = X_train[idx]
                self.save_imgs(epoch, imgs)

    def save_imgs(self, epoch, imgs):
        r, c = 5, 5

        encoded_imgs = self.encoder.predict(imgs)
        gen_imgs = self.decoder.predict(encoded_imgs)

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                # axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].plot(gen_imgs[cnt])
                axs[i,j].axis('on')
                cnt += 1
        fig.savefig("../AAE/Plots/Cl_%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "../AAE/Saved_model/%s.json" % model_name
            weights_path = "../AAE/Saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "aae_generator")
        save(self.discriminator, "aae_discriminator")


if __name__ == '__main__':

    aae = AdversarialAutoencoder()
    num_epochs = 1000 #20000
    batch_size = 8	 #32
    save_interval = 200 # 200
    totalFiles = 256
    TestFiles = 32
    
		    
    train_path = '../Cl_data/Data/LatinCl_'+str(totalFiles)+'.npy'
    train_target_path =  '../Cl_data/Data/LatinPara5_'+str(totalFiles)+'.npy'
    test_path = '../Cl_data/Data/LatinCl_'+str(TestFiles)+'.npy'
    test_target_path =  '../Cl_data/Data/LatinPara5_'+str(TestFiles)+'.npy'




    aae.train(epochs=num_epochs, batch_size=batch_size, save_interval=save_interval)
