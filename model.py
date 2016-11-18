from keras.models import Model
from keras.models import load_model
from keras.layers import Convolution1D, AtrousConvolution1D, Input
from keras.layers import merge

class WavenetModel:
    def __init__(self,
                 filename="model.h5",
                 numBlocks=2,
                 numLayers=8,
                 numFilters=32,
                 filterSize=2,
                 frameSize=2048):
        self.numBlocks = numBlocks
        self.numLayers = numLayers
        self.numFilters = numFilters
        self.filterSize = filterSize
        self.frameSize = frameSize
        self.model = self._getKerasModel()

    # 2.3: Unit activation function
    def _activation(self, data, dilation):
        tanh = AtrousConvolution(self.numFilters,
                                 self.filterSize,
                                 atrous_rate=dilation,
                                 activation='tanh')
        sigm = AtrousConvolution(self.numFilters,
                                 self.filterSize,
                                 atrous_rate=dilation,
                                 activation='sigmoid')
        return merge([tanh(data), sigm(data)], mode='mul')

    def _getBlock(self, data, dilation):
        result = self._activation(data, dilation)
        # 2.4: Residaual and Skip connections
        skip = Convolution1D(1, 1)(result)
        residual = merge([data, skip], mode='sum')
        return residual, skip

    # Returns a Keras Model with appropriate convolution layers.
    # I kinda just follow what the figure looks like in Section 2.4
    def _getKerasModel(self):
        input_ = Input(shape=(self.frameSize, 1))
        # Initial convolution (what the eff is a causal convolution?):
        residual = AtrousConvolution(self.numFilters,
                                self.filterSize,
                                atrous_rate=1)(input_)

        # Convolutional layers: calculating residual blocks
        skips = [residual]
        for i in range(numLayers):
            residual, skip = self._getBlock(residual, 2 ** i)
            skips.append(skip)
    
        # Skip connection output: (from figure):
        #   SUM => RELU => 1x1 CONV => RELU => 1x1 CONV => SOFTMAX
        result = merge(skips, mode='sum')
        result = Activation('relu')(result)
        result = Convolution1D(1, 1, activation='relu')
        result = Convolution1D(1, 1)
        result = Activation('softmax')(result)

        model = Model(input=input_, output=result)
        # TODO (sydli): What do we use for loss here??
        model.compile(optimizer='sgd', loss='mse')
        modal.summary()
        return model

    def train(self, data, save=True):
        self.model.fit_generator(data, samples_per_epoch=2000, nb_epoch=1000)
        if save: self.model.save(self.filename)

    def load(self):
        self.model = load_model(self.filename)

# 2.5: Conditional wavenets
class ConditionedWavenetModel(WavenetModel):
    pass

if __name__=="__main__":
    model = WavenetModel()

