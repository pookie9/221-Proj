from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.models import load_model
from keras.layers import Convolution1D, AtrousConvolution1D, Input, Activation, Dense, Flatten, Merge
from keras.layers import merge
from keras import metrics

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
                 numLayers=6,
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
        self.model = self._getKerasModel()

    # 2.3: Unit activation function
    def _activation(self, data, dilation):
        tanh = AtrousConvolution1D(self.numFilters,
                                 self.filterSize,
                                 atrous_rate=dilation,
                                 border_mode='same',
                                 activation='tanh')(data)
        sigm = AtrousConvolution1D(self.numFilters,
                                 self.filterSize,
                                 atrous_rate=dilation,
                                 border_mode='same',
                                 activation='sigmoid')(data)
        return merge([tanh, sigm], mode='mul')

    def _getBlock(self, data, dilation):
        result = self._activation(data, dilation)
        # 2.4: Residaual and Skip connections
        skip = Convolution1D(self.numFilters, 1, border_mode='same')(result)
        residual = merge([data, skip], mode='sum')
        return residual, skip

    # Returns a Keras Model with appropriate convolution layers.
    # I kinda just follow what the figure looks like in Section 2.4
    def _getKerasModel(self):
        input_ = Input(shape=(self.frameSize, 256))
        residual = input_
        print residual
        residual = AtrousConvolution1D(self.numFilters, 2, atrous_rate=1, border_mode='same')(residual)
        print residual
        # Convolutional layers: calculating residual blocks
        # Skip connections are used for regularization / prevent overfitting
        skips = []
        for i in range(self.numLayers):
            residual, skip = self._getBlock(residual, 2 ** i)
            skips.append(skip)
    
        # Skip connection output: (from figure):
        #   SUM => RELU => 1x1 CONV => RELU => 1x1 CONV => SOFTMAX
        result = Merge(mode='sum')(skips)
        result = Activation('relu')(result)
        result = Convolution1D(1, 1, activation='relu', border_mode='same')(result)
        result = Convolution1D(1, 1, border_mode='same')(result)
        # result = Activation('softmax')(result)

        # # I saw someone else do this to flatten the dimensionality => 256 values :D
        # # Smart shit
        result = Flatten()(result)
        result = Dense(256, activation='softmax')(result)

        model = Model(input=input_, output=result)
        model.compile(optimizer='sgd', loss='categorical_crossentropy',
                metrics = [metrics.mean_squared_error, metrics.categorical_accuracy])
        model.summary()
        return model

    ######## Public member functions

    # Train this model on supplied data object.
    def train(self):
        # X, y = self.data.get()
        print "Training on data..."
        checkpoint = ModelCheckpoint(filepath="weights.{epoch:03d}.hdf5",
                verbose=1, save_best_only=False, save_weights_only=False, mode='auto')
        self.model.fit_generator(self.data.getGenerator(),
                samples_per_epoch=5000,
                nb_epoch=self.numEpochs,
                callbacks=[checkpoint])
        print "Finished Training on data!"

    # Generate |numSeconds| from trained model.
    def generate(self, numSeconds, filename="generated.wav"):
        print "Generating music..."
        numSamples = numSeconds * self.data.sampleRate
        seed = self.data.getSeed()
        # Unroll samples (for some reason it's 2d array.. should probably debug)
        samples = list([s[0] for s in seed]) 
        print samples
        i = 0
        eye = np.eye(256)
        while len(samples) < numSamples:
            sys.stdout.write("\r%d/%d" % (i, numSamples))
            sys.stdout.flush()
            input_ = np.asarray(samples[i:i+self.frameSize]).reshape((1, self.frameSize, 1))
            result = self.model.predict(input_)
            result /= result.sum().astype(float) # normalize
            result = result.reshape(256)
            sample = np.random.choice(range(256), p=result)
            samples.append(eye[sample])
            i += 1
        samples = [util.inverse_mulaw(s) for s in samples]
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

