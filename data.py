import librosa
import numpy as np
import random
import os

import util

class AudioData(object):
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
        

class AudioDataWithGlobal(AudioData):

    def __init__(self,
                 sampleRate,
                 frameSize,
                 frameShift,
                 filename,
                 globalVar
                 ):
        super(AudioDataWithGlobal,self).__init__(sampleRate,frameSize,frameShift,filename=filename)
        self.globalVar=globalVar
    

    def get(self):
        training,targets=super(AudioDataWithGlobal,self).get()
        training=np.dstack((training, np.zeros((training.shape[0],training.shape[1],1))+self.globalVar))
        targets=targets.reshape((targets.shape[0],targets.shape[1],1))
        targets=np.dstack((targets,np.zeros((targets.shape[0],targets.shape[1],1))+self.globalVar))
        return training,targets


class AudioCollection(AudioData):

    def __init__(self,
                 sampleRate,
                 frameSize,
                 frameShift,
                 filenames,
                 globalVars
                 ):

        self.frameSize = frameSize
        self.frameShift = frameShift
        self.sampleRate = sampleRate
        self.training = None
        self.audio = []
        self.audioFiles = []
        self.audiodata=[AudioDataWithGlobal(sampleRate,frameSize,frameShift,filename,globalVars) for filename,globalVars in zip(filenames,globalVars)]
        self.globalVars=globalVars
    
    def getSeed(self,globalVar):
        valid_files=[i for i in range(len(self.globalVars)) if self.globalVars[i]==globalVar]
        print valid_files
        return self.audiodata[random.sample(valid_files,1)[0]].getSeed()


    def get(self):
        training=[]
        targets=[]
        for audiodatum in self.audiodata:
            train,targ=audiodatum.get()
            training.extend(train)
            targets.extend(targ)
        return training, targets

if __name__=="__main__":
#    data = AudioData(sampleRate=8000, frameSize=2048, frameShift=128, filename="piano.wav")
#    X, y = data.get()
    data = AudioCollection(8000, 2048, 128, ["piano.wav","orchestra.wav"],[1,0])
    X,y=data.get()
    print len(X)
    print X[0].shape
    print y[0].shape
    print len(y)
    
