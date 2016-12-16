from model import WavenetModel
from data import AudioData

if __name__=="__main__":
    data = AudioData(sampleRate=4410,
            frameSize=5119, frameShift=128,
            filename="piano.wav")
    model = WavenetModel(data)
    # model.model.load_weights("weights.010.hdf5")
    model.train()
    # model.save("model.h5")

    generated = model.generate(1)
    with open("generated", "w+") as f:
        for item in generated: f.write("%s\n" % item)
