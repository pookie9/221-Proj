from model import WavenetModel
from data import *

if __name__=="__main__":
    data = AudioCollection(8000, 2048, 128, ["piano.wav","orchestra.wav"],[1,0])
    model = ConditionedWavenetModel(data, frameSize=64)
    model.train()
    model.save("model.h5")
    model.load("model.h5")

    generated = model.generate(24000)
    with open("generated", "w+") as f:
        for item in generated: f.write("%s\n" % item)
