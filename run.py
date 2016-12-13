from model import WavenetModel
from data import AudioData

if __name__=="__main__":
    data = AudioData(sampleRate=4410,
            frameSize=2048, frameShift=128,
            filename="orchestra.wav")
    model = WavenetModel(data)
    model.model.load_weights("weights.100.hdf5")
    # model.train()
    model.save("model.h5")

    generated = model.generate(1)
    with open("generated", "w+") as f:
        for item in generated: f.write("%s\n" % item)
