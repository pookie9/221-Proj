from model import WavenetModel
from data import AudioData

if __name__=="__main__":
    data = AudioData(sampleRate=8000, frameSize=64, frameShift=2, filename="piano.wav")
    model = WavenetModel(data)
    model.load("model.h5")
    # model.train()
    # model.save("model.h5")

    generated = model.generate(24000)
    with open("generated", "w+") as f:
        for item in generated: f.write("%s\n" % item)
