import librosa
import numpy as np
import os

class AudioData:
    def __init__(self,
                 sampleRate,
                 frameSize,
                 frameShift,
                 filename=None, directory=None):
        self.frameSize = frameSize
        self.frameShift = frameShift
        self.sampleRate = sampleRate
        self.audio = []
        self.audioFiles = []
        if directory is not None:
            self.loadDirectory(directory)
        elif filename is not None:
            self.addFile(filename)

    def loadDirectory(self, directory, ext="wav"):
        self.audio = []
        self.audioFiles = [f for f in os.listdir(directory) if f.endswith(".%s" % ext)]

    def addFile(self, filename):
        self.audioFiles.append(filename)

    def _loadAudio(self):
        audio = []
        print "Loading audio from files..."
        for filename in self.audioFiles:
            fileAudio, _ = librosa.load(filename, sr=self.sampleRate, mono=True)
            fileAudio = fileAudio.reshape(-1, 1)
            audio += fileAudio
        print "Finished loading audio."
        return np.asarray(audio)

    def get(self):
        audio = self._loadAudio()
        training = []
        targets = []
        for i in range(len(audio) - self.frameSize - 1, self.frameShift):
            slice_ = audio[i:i+self.frameSize]
            # Quantize target according to part 2.2
            target = audio[i + self.frameSize + 1]
            target = int(np.sign(target) * (np.log(1 + 255*abs(target)) / np.log(1+255)))
            training.append(slice_.reshape(frameSize, 1))
            print target
            targets.append(target)
        return training, targets

if __name__=="__main__":
    data = AudioData(sampleRate=8000, frameSize=2048, frameShift=12, filename="piano.wav")
    data.get()
