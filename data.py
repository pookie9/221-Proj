import librosa
import numpy as np
import random
import os

import util

class AudioData:
    def __init__(self,
                 sampleRate,
                 frameSize,
                 frameShift,
                 filename=None, directory=None):
        self.frameSize = frameSize
        self.frameShift = frameShift
        self.sampleRate = sampleRate
        self.training = None
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
        audio = np.asarray([])
        print "Loading audio from files..."
        for filename in self.audioFiles:
            fileAudio, _ = librosa.load(filename, sr=self.sampleRate, mono=True)
            audio = fileAudio
        print "Finished loading audio."

        # normalize
        audio = audio.astype(float)
        audio = audio - audio.min()
        audio = audio / (audio.max() - audio.min())
        audio = (audio - 0.5) * 2
        return audio

    def get(self):
        if self.training is not None:
            return self.training, self.targets
        audio = self._loadAudio()
        audio = audio[0:len(audio)/2]
        training = []
        targets = []
        eye = np.eye(256)
        for i in range(0, len(audio) - self.frameSize - 1, self.frameShift):
            slice_ = audio[i:i + self.frameSize]
            target = audio[i + self.frameSize + 1]
            # Quantize target according to part 2.2
            target = util.mulaw(target)
            training.append(slice_.reshape(self.frameSize, 1))
            targets.append(eye[target])
        self.training = np.asarray(training)
        self.targets = np.asarray(targets)
        return np.asarray(training), np.asarray(targets)

    def getSeed(self):
        self.get()
        i = random.randint(0, len(self.training)-1)
        return self.training[i]

if __name__=="__main__":
    data = AudioData(sampleRate=8000, frameSize=2048, frameShift=128, filename="piano.wav")
    X, y = data.get()
