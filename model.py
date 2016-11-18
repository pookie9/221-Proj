from keras.layers import Convolution1D, AtrousConvolution1D
from keras.layers import merge

class WavenetModel:
    def __init__(self,
                 numBlocks=2,
                 numLayers=8,
                 numFilters=32,
                 filterSize=2):
        self.numBlocks = numBlocks
        self.numLayers = numLayers
        self.numFilters = numFilters
        self.filterSize = filterSize

    # 2.3: Unit activation function
    def _activation(data, dilation):
        tanh = AtrousConvolution(self.numFilters,
                                 self.filterSize,
                                 atrous_rate=dilation,
                                 activation='tanh')
        sigm = AtrousConvolution(self.numFilters,
                                 self.filterSize,
                                 atrous_rate=dilation,
                                 activation='sigmoid')
        return merge([tanh(data), sigm(data)], mode='mul')

    def getBlock(data, dilation):
        result = self._activation(data, dilation)
        # 2.4: Residaual and Skip connections
        skip = Convolution1D(1, 1)(result)
        residual = merge([data, skip], mode='sum')
        return residual, skip

    # Returns a Keras Model with appropriate convolution layers.
    # I kinda just follow what the figure looks like in Section 2.4
    def get():
        input_ = Input(shape=(input_size, 1))
        # Initial convolution (what the eff is a causal convolution?):
        residual = AtrousConvolution(self.numFilters,
                                self.filterSize,
                                atrous_rate=1)(input_)

        # Convolutional layers: calculating residual blocks
        skips = [residual]
        for i in range(numLayers):
            residual, skip = self.getBlock(residual, 2 ** i)
            skips.append(skip)
    
        # Skip connection output: (from figure):
        #   SUM => RELU => 1x1 CONV => RELU => 1x1 CONV => SOFTMAX
        result = merge(skips, mode='sum')
        result = Activation('relu')(result)
        result = Convolution1D(1, 1, activation='relu')
        result = Convolution1D(1, 1)
        result = Activation('softmax')(result)

        return Model(input=input_, output=result)

# 2.5: Conditional wavenets
class ConditionedWavenetModel(WavenetModel):
    pass

