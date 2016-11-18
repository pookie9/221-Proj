from keras.models import Model
from keras.models import load_model
from keras.layers import Convolution1D, AtrousConvolution1D, Input, Activation, Dense, Flatten
from keras.layers import merge

from data import AudioData

class WavenetModel:
    def __init__(self,
                 data,
                 numBlocks=2,
                 numLayers=8,
                 numFilters=32,
                 filterSize=2,
                 frameSize=256):
        self.data = data
        self.numBlocks = numBlocks
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

    def train(self, save=True):
        X, y = self.data.get()
        print "Training on data..."
        self.model.fit(X, y)
        print "Finished Training on data!"

    def generate(numSamples):
        seed = self.data.getSeed()
        samples = seed
        i = 0
        while len(samples) < numSamples:
            input_ = samples[i:i+self.frameSize]
            result = self.model.predict(input_)
            result /= result.sum().astype(float) # normalize
            sample = np.random.choice(range(256), result)
            # TODO (sydli): reconstruct audio signal from discretized sample
            samples.append(sample)
            i += 1
        return samples

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = load_model(filename)
        self.filename = filename

# 2.5: TODO (sydli): Conditional wavenets-- we can condition globally on things
#      like speaker identity, or locally on things like phonemes.
class ConditionedWavenetModel(WavenetModel):
    pass
