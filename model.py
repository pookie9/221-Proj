from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.models import load_model
from keras.layers import Convolution1D, AtrousConvolution1D, Input, Activation, Dense, Flatten, Merge
from keras.layers import merge
from keras import metrics

import matplotlib.pyplot as plt

import librosa
import numpy as np
import sys

from data import AudioData
import util

# Wrapper around conv neural net constructed after wavenet paper.
# uses Keras primitives.
class WavenetModel:
    # Usage:
    #   data          An AudioData object initialized with training audio data
    #   numLayers     The number of dilated convolution layers in our model.
    #   numFilters    The number of filters to apply per layer. Affects
    #                 dimensionality of hidden layer outputs.
    #   filterSize    The size of a filter's spatial "neighborhood".
    # 
    #   train() /  generate()
    #     train the Keras model on the supplied data, then generate
    #     a number of samples using the trained model weights.
    #   save() / load()
    #     push/pull models to/from disk.
    def __init__(self,
                 data,
                 numEpochs=1000,
                 numLayers=7,
                 numFilters=16,
                 filterSize=2):
                 # TODO (sydli): Calculate frame size + shift from 
                 # receptive field so we don't do unnecessary computation
        self.data = data
        self.numEpochs = numEpochs
        self.numLayers = numLayers
        self.numFilters = numFilters
        self.filterSize = filterSize
        self.frameSize = data.frameSize
        self.numQuantize = data.numQuantize
        # Construct model
        self.model = self._getKerasModel()

    # 2.3: Unit activation function
    def _activation(self, data, dilation):
        tanh = AtrousConvolution1D(self.numFilters,
                                 self.filterSize,
                                 atrous_rate=dilation,
                                 border_mode='valid',
                                 causal=True,
                                 bias=False,
                                 activation='tanh')(data)
        sigm = AtrousConvolution1D(self.numFilters,
                                 self.filterSize,
                                 atrous_rate=dilation,
                                 border_mode='valid',
                                 causal=True,
                                 bias=False,
                                 activation='sigmoid')(data)
        return merge([tanh, sigm], mode='mul')

    def _getBlock(self, data, dilation):
        result = self._activation(data, dilation)
        # 2.4: Residaual and Skip connections
        skip = Convolution1D(self.numFilters, 1, bias=False, border_mode='same')(result)
        residual = merge([data, skip], mode='sum')
        return residual, skip

    # Returns a Keras Model with appropriate convolution layers.
    # I kinda just follow what the figure looks like in Section 2.4
    def _getKerasModel(self):
        input_ = Input(shape=(self.frameSize, self.numQuantize))
        residual = AtrousConvolution1D(self.numFilters, 2, atrous_rate=1, border_mode='valid', causal=True)(input_)
        # Convolutional layers: calculating residual blocks
        # Skip connections are used for regularization / prevent overfitting
        skips = []
        for i in range(self.numLayers+1):
            residual, skip = self._getBlock(residual, 2 ** i)
	    skips.append(skip)
    
        # Skip connection output: (from figure):
        #   SUM => RELU => 1x1 CONV => RELU => 1x1 CONV => SOFTMAX
        result = Merge(mode='sum')(skips)
        result = Activation('relu')(result)
        result = Convolution1D(self.numQuantize, 1, activation='relu', border_mode='same')(result)
        result = Convolution1D(self.numQuantize, 1, border_mode='same')(result)
        result = Activation('softmax')(result)

        model = Model(input=input_, output=result)
        model.compile(optimizer='sgd', loss='categorical_crossentropy',
                metrics = [metrics.categorical_mean_squared_error, metrics.categorical_accuracy])
        model.summary()
        return model

    ######## Public member functions

    # Train this model on supplied data object.
    def train(self):
        print "Training on data..."
        checkpoint = ModelCheckpoint(filepath="weights.{epoch:03d}.hdf5",
                verbose=1, save_best_only=False, save_weights_only=False, mode='auto')
        self.model.fit_generator(self.data.getGenerator(),
                samples_per_epoch=5000,
                nb_epoch=self.numEpochs,
                callbacks=[checkpoint])
        print "Finished Training on data!"

    # Generate |numSeconds| from trained model.
    def generate(self, numSeconds, filename="generated"):
        numSamples = numSeconds * self.data.sampleRate
        seed = self.data.getSeed()
        print "Generating music..."
        samples = seed.reshape(self.frameSize, self.numQuantize)
        numSamples = len(samples) + 100#numSeconds * self.data.sampleRate + len(samples)
        i = 0
        while len(samples) < numSamples:
            print("\r%d/%d" % (i, numSamples))
            # sys.stdout.flush()
            input_ = np.asarray(samples[i:i+self.frameSize]).reshape((1, self.frameSize, self.numQuantize))
            result = self.model.predict(input_).reshape(self.frameSize, self.numQuantize)
            result = result[-1]
            result /= result.sum().astype(float) # normalize
            result = result.reshape(self.numQuantize)
            plt.plot(result)
            # plt.savefig("%d_dist.png" % i)
            plt.clf()
            sample = np.argmax(result)#np.random.choice(range(self.numQuantize), p=result)
            print sample
            samples = np.append(samples, [util.one_hot(sample, self.numQuantize)], axis=0)
            i += 1
        quantized = [list(s).index(1) for s in samples]
        samples = [util.inverse_mulaw(s, self.numQuantize-1) for s in samples]
        print samples
        print "Writing to wav..."
        librosa.output.write_wav(filename + ".wav", np.asarray(samples), self.data.sampleRate)
        plt.plot(quantized)
        plt.savefig(filename + "_quantized.png")
        plt.clf()
        plt.plot(samples)
        plt.savefig(filename + "_audio.png")
        plt.clf()
        return samples

    # Saves model to filename.
    def save(self, filename):
        self.model.save(filename)

    # Loads model from filename.
    def load(self, filename):
        self.model = load_model(filename)
        self.filename = filename

# 2.5: TODO (sydli): Conditional wavenets-- we can condition globally on things
#      like speaker identity, or locally on things like phonemes.
class ConditionedWavenetModel(WavenetModel):
    
    def __init__(self,
                 data,
                 speakers,
                 numLayers=6,
                 numFilters=32,
                 filterSize=2,
                 # TODO (sydli): Calculate frame size + shift from 
                 # receptive field so we don't do unnecessary computation
                 frameSize=64):
        self.speakers = speakers
        super(WavenetModel,self).__init__(data, numLayers,numFilters,filterSize)
        print type(self.data)
        exit(1)
    def _activation(self, data, dilation):
        tanh = AtrousConvolution1D(self.numFilters,
                                 self.filterSize,
                                 atrous_rate=dilation,
                                 border_mode='same',
                                 activation='tanh')
        sigm = AtrousConvolution1D(self.numFilters,
                                 self.filterSize,
                                 atrous_rate=dilation,
                                 border_mode='same',
                                 activation='sigmoid')
        return merge([tanh(data), sigm(data)], mode='mul')

