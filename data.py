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

        # normalize & quantize according to part 2.2
        # the audio array is a huge memory hog so we'll do it carefully...
        min_ = audio[0]
        max_ = audio[0]
        for i in xrange(len(audio)):
            if audio[i] > max_:
                max_ = audio[i]
            if audio[i] < min_:
                min_ = audio[i]

        for i in xrange(len(audio)):
            audio[i] = audio[i].astype(float)
            audio[i] = audio[i] - min_
            audio[i] = audio[i] / (max_ - min_)
            audio[i] = (audio[i] - 0.5) * 2
        return audio

    # Generates random samples
    def getGenerator(self):
        audio = self._loadAudio()
        training = []
        targets = []
        while True:
            i = random.randint(0, len(audio) - self.frameSize - 1)
            slice_ = np.asarray([util.mulaw(a) for a in audio[i:i + self.frameSize]])
            slice_ = slice_.reshape(1, self.frameSize, 256)
            print sum(slice_[0][0])
            target = np.asarray(util.mulaw(audio[i + self.frameSize + 1])).reshape(1, 256)
            yield slice_.reshape(1, self.frameSize, 256), target

    def getSeed(self):
        return next(self.getGenerator())[0]

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


class AudioConditioned(AudioData):
    
    def __init__(self,sampleRate,frameSize,frameShift,filename):
        self.frameSize=frameSize
        self.frameShift=frameShift
        self.sampleRate=sampleRate
        self.training=None
        self.audio=[]
        self.audiodata=AudioData(sampleRate,frameSize,frameShift,filename='data/wav/'+filename+'.wav')._loadAudio()
        self.labels=np.fromfile('data/binary_label_norm/'+filename+'.lab', dtype=np.float32, count=-1)
        self.labels=self.labels.reshape((self.labels.shape[0]/425,425))
        self.audiodata=np.reshape(self.audiodata,(self.audiodata.shape[0],1))
        samples_per_frame=self.audiodata.shape[0]/self.labels.shape[0]
        self.labels=np.repeat(self.labels,samples_per_frame,0)
        #Should be pretty close to each other, so now just trim to be exact length
        if self.labels.shape[0]>self.audiodata.shape[0]:
            self.labels=self.labels[:self.audiodata.shape[0],:]
        else:
            self.audiodata=self.audiodata[:self.labels.shape[0],:]
        print self.labels.shape
        print self.audiodata.shape 
if __name__=="__main__":

    data = AudioConditioned(sampleRate=8000, frameSize=2048, frameShift=128, filename="arctic_a0001")
    
