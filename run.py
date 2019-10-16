import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, concatenate
from tensorflow.keras import Input, Model


class Network:

	def __init__(self):

		self.regL1 = None
		self.train_input = None
		self.Input = Input(shape=[1600, 256, 3])
		self.conv_net_1 = self.conv1(self.Input)
		self.AE1_output = self.autoencoder1(self.conv_net_1)
		self.AE2_output = self.autoencoder2(self.conv_net_1)
		self.AE3_output = self.autoencoder3(self.conv_net_1)
		self.AE4_output = self.autoencoder4(self.conv_net_1)
		self.layer2 = concatenate(
			[self.AE1_output, self.AE2_output, self.AE3_output, self.AE4_output, self.conv_net_1])
		self.output = self.conv2(self.layer2)
		self.model = Model(self.Input, self.output)
		self.model.compile(
			optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=self.loss)
		self.model.summary()

	def conv1(self, x):
		x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
		x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
		x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
		return x

	def autoencoder1(self, x):
		x = Conv2D(16, (2, 2), (2, 2), activation='relu', padding='same')(x)
		x = Conv2DTranspose(8, (2, 2), (2, 2), activation='relu', padding='same')(x)
		return x

	def autoencoder2(self, x):
		x = Conv2D(16, (4, 4), (4, 4), activation='relu', padding='same')(x)
		x = Conv2DTranspose(8, (4, 4), (4, 4), activation='relu', padding='same')(x)
		return x

	def autoencoder3(self, x):
		x = Conv2D(16, (8, 8), (8, 8), activation='relu', padding='same')(x)
		x = Conv2DTranspose(8, (8, 8), (8, 8), activation='relu', padding='same')(x)
		return x

	def autoencoder4(self, x):
		x = Conv2D(16, (16, 16), (16, 16), activation='relu', padding='same')(x)
		x = Conv2DTranspose(8, (16, 16), (16, 16),
		                    activation='relu', padding='same')(x)
		return x

	def conv2(self, x):
		x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
		x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
		x = Conv2D(8, (7, 7), activation='relu', padding='same')(x)
		x = Conv2D(5, (3, 3), activation='sigmoid', padding='same')(x)
		return x

	def dice(self, x, y):
		num = tf.math.scalar_mul(2, tf.norm(tf.math.multiply(x, y)))
		den = tf.math.add(tf.math.add(tf.norm(x), tf.norm(y)), 1e-6)
		return tf.math.divide(num, den)

	def loss(self, x, y):
		dice_coeff = tf.math.negative(tf.math.log(self.dice(x, y)))
		return dice_coeff

	def train_batch(self, input, output):
		history = self.model.train_on_batch(input, output)
		print('Loss:{}'.format(history))

	def predict(self, input):
		output = self.model.predict(input)
		return output


class DataLoader:

    def __init__(self):
        self.train = pd.read_csv('train.csv').fillna('Nan')
        self.test = pd.read_csv('sample_submission.csv')
        self.model = Network()
        self.epoch = 100
        self.train_path = 'train/'
        self.test_path = 'test/'
        self.let_train()

    def mask2rle(self, img):
        '''
        img: numpy array, 1 - mask, 0 - background
        Returns run length as string formated
        '''
        pixels = img.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

    def rle2mask(self, mask_rle, shape=(1600, 256)):
        '''
        mask_rle: run-length as string formated (start length)
        shape: (width,height) of array to return 
        Returns numpy array, 1 - mask, 0 - background
        '''
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int)
                           for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape)

    def let_train(self):
        for epoch in range(self.epoch):
            print('Epoch:{}/{}'.format(epoch+1, self.epoch))
            for i in range(self.train.shape[0]//4):
                img = Image.open('{}{}'.format(
                    self.train_path, self.train.iat[4*i, 0].split('_')[0]))
                output = np.zeros([1600, 256, 5])
                for j in range(4):
                    if self.train.iat[4*i+j, 1] != 'Nan':
                        output[:, :, j] = self.rle2mask(
                            self.train.iat[4*i+j, 1])
                output[:, :, 4] = 1 - np.clip(output[:, :, 0] + output[:, :, 1] +
                                              output[:, :, 2] + output[:, :, 3], a_min=0, a_max=1)
                input = np.array(img).reshape([1600, 256, 3])
                self.model.train_batch(np.expand_dims(
                    input, axis=0), np.expand_dims(output, axis=0))
                del input, output, img
            if epoch and epoch % 100 == 0:
                self.let_test(epoch)
                self.model.model.save('Model{}.h5'.format(epoch))

    def let_test(self, epoch=0):
        for i in range(self.test.shape[0]//4):
            img = Image.open('{}{}'.format(
            	self.train_path, self.train.iat[4*i, 0].split('_')[0]))
            input = np.asarray(img).reshape([1600, 256, 3])
            ans = self.model.predict(np.expand_dims(input, axis=0))
            output = np.argmax(ans, axis=-1)
            out = [1, 2, 3, 4]
            out[0] = (output == 0).astype('uint8')
            out[1] = (output == 1).astype('uint8')
            out[2] = (output == 2).astype('uint8')
            out[3] = (output == 3).astype('uint8')
            for j in range(4):
                self.test.iat[4*i+j, 1] = self.mask2rle(out[j])
            del input, output, img, out
        self.test.to_csv('answers{}.csv'.format(epoch), index=None)


DataLoader()
