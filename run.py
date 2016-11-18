from model import WavenetModel
from data import AudioData

if __name__=="__main__":
    data = AudioData(sampleRate=8000, frameSize=256, frameShift=2, filename="piano.wav")
    model = WavenetModel(data)
    model.train()
    model.save("model.h5")

    generated = model.generate(32000)
    with open("generated", "rw+") as f:
        for item in generated: f.write("%s\n" % item)
