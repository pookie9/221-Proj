from keras.models import Model
from keras.models import load_model
from keras.layers import Convolution1D, AtrousConvolution1D, Input, Activation, Dense, Flatten
from keras.layers import merge

import librosa
import numpy as np

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
                 numLayers=6,
                 numFilters=32,
                 filterSize=2,
                 # TODO (sydli): Calculate frame size + shift from 
                 # receptive field so we don't do unnecessary computation
                 frameSize=64):
        self.data = data
        self.numLayers = numLayers
        self.numFilters = numFilters
        self.filterSize = filterSize
        self.frameSize = frameSize
        self.model = self._getKerasModel()

    # 2.3: Unit activation function
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

    def _getBlock(self, data, dilation):
        result = self._activation(data, dilation)
        # 2.4: Residaual and Skip connections
        skip = Convolution1D(1, 1, border_mode='same')(result)
        residual = merge([data, skip], mode='sum')
        return residual, skip

    # Returns a Keras Model with appropriate convolution layers.
    # I kinda just follow what the figure looks like in Section 2.4
    def _getKerasModel(self):
        input_ = Input(shape=(self.frameSize, 1))
        # TODO (sydli): Initial convolution?
        residual = input_

        # Convolutional layers: calculating residual blocks
        skips = [residual]
        for i in range(self.numLayers):
            residual, skip = self._getBlock(residual, 2 ** i)
            skips.append(skip)
    
        # Skip connection output: (from figure):
        #   SUM => RELU => 1x1 CONV => RELU => 1x1 CONV => SOFTMAX
        result = merge(skips, mode='sum')
        result = Activation('relu')(result)
        result = Convolution1D(1, 1, activation='relu', border_mode='same')(result)
        result = Convolution1D(1, 1, border_mode='same')(result)

        # I saw someone else do this to flatten the dimensionality => 256 values :D
        # Smart shit
        result = Flatten()(result)
        result = Dense(256, activation='softmax')(result)

        model = Model(input=input_, output=result)
        # TODO (sydli): What do we use for loss here??
        model.compile(optimizer='sgd', loss='categorical_crossentropy')
        model.summary()
        return model

    ######## Public member functions

    # Train this model on supplied data object.
    def train(self):
        X, y = self.data.get()
        print "Training on data..."
        self.model.fit(X, y)
        print "Finished Training on data!"

    # Generate |numSamples| from trained model.
    def generate(self, numSamples, filename="generated.wav"):
        seed = self.data.getSeed()
        # Unroll samples (for some reason it's 2d array.. should probably debug)
        samples = list([s[0] for s in seed]) 
        print samples
        i = 0
        while len(samples) < numSamples:
            input_ = np.asarray(samples[i:i+self.frameSize]).reshape((1, self.frameSize, 1))
            result = self.model.predict(input_)
            result /= result.sum().astype(float) # normalize
            result = result.reshape(256)
            sample = np.random.choice(range(256), p=result)
            samples.append(util.inverse_mulaw(sample))
            i += 1
        print "Writing to wav..."
        librosa.output.write_wav(filename, np.asarray(samples), self.data.sampleRate)
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
    pass
