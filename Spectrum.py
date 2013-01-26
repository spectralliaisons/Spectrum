import pyaudio
import math
import wave
import audioread
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
import warnings
import struct

###
### Wes Jackson Oct 2012
### 
### Consistent analysis for: mp3, wav, aiff
###

class Analyze:
    def __init__(self, fileName, maxFreq=12000):
        
        # run 1D real fft
        self.analyze(fileName, maxFreq)  
        
    #########################   
    # Spectral Analysis
    #########################
    
    def analyze(self, fileName, maxFreq):
        
        print "Running 1D real fft. . ."

        self.clear()
        self.fMax = maxFreq
        
        # get analysis for each chunk of audio
        with audioread.audio_open(fileName) as self.audata:
            for self.buf in self.audata:
                self.fftBuf()
        
        self.generalAnalysis()

    def clear(self):
        self.allF0 = []
        self.allEnergy = []
        self.allCentroid = []
        self.allSkewness = []
        self.allKurtosis = []
        self.FFT = []
        self.freqRange = []
    
    #########################   
    # Run FFT for a chunk of audio and compile analysis
    #########################

    def fftBuf(self):
        
        self.updateAudioData()

        #indata = np.array(wave.struct.unpack("%dh"%(len(data)/self.swidth), data))*self.window
        fmt = "%dh" % self.chunk # signed 2-byte shorts
        indata = np.array(struct.unpack(fmt, self.buf)) * self.window
        
        realFFT = abs(np.fft.rfft(indata)[0:self.freqToIndex(self.fMax)])**2   # 1D real fft -- squared

        # normalization seems to cause problems
        normalizedFFT = realFFT / np.amax(realFFT) # normalized to greatest value
            
        self.FFT.append(realFFT)

        self.updateAnalysis(realFFT)

    def updateAnalysis(self, fft):

        ## Calculate Energy
        energy = fft.sum()
        self.allEnergy.append(energy)
            
        ### Calculate Fundamental Freq
        f0 = self.calculateF0(fft)    
        self.allF0.append(f0)
            
        ### Calculate Spectral Centroid
        centroid = self.calculateSpectralCentroid(fft)
        self.allCentroid.append(centroid)
            
        ### Calculate Skewness
        skewness = stats.skew(fft)
        self.allSkewness.append(skewness)
            
        ### Calculate Kurtosis
        kurtosis = stats.kurtosis(fft)
        self.allKurtosis.append(kurtosis)

    def generalAnalysis(self):

        self.loseLast();

        # General Analysis for whole sound

        # normalize energy across entire sound
        self.allEnergy = self.allEnergy/np.amax(self.allEnergy)

        # Unweighted mean across entire sound file
        self.MeanF0 = np.mean(self.allF0)
        self.MeanEnergy = np.mean(self.allEnergy)
        self.MeanSpectralCentroid = np.mean(self.allCentroid)
        self.MeanSkewness = np.mean(self.allSkewness)
        self.MeanKurtosis = np.mean(self.allKurtosis)

        ### !!! these values change depending on whether fft is normalized

        # weighted mean by energy for e/a chunk
        self.WeightedF0 = np.average(self.allF0, weights=self.allEnergy)
        self.WeightedSpectralCentroid = np.average(self.allCentroid, weights=self.allEnergy)
        self.WeightedSkewness = np.average(self.allSkewness, weights=self.allEnergy)
        self.WeightedKurtosis = np.average(self.allKurtosis, weights=self.allEnergy)

        print "Done!"
        print "Unweighted, weighted mean:"
        print ">> F0: %d, %d" % (self.MeanF0, self.WeightedF0)
        print ">> Energy: %f" % self.MeanEnergy
        print ">> Spectral Centroid: %d, %d" % (self.MeanSpectralCentroid, self.WeightedSpectralCentroid)
        print ">> Skewness: %d, %d" % (self.MeanSkewness, self.WeightedSkewness)
        print ">> Kurtosis: %d, %d" % (self.MeanKurtosis, self.WeightedKurtosis)

    def updateAudioData(self):
        self.swidth = 2 # 16-bit ^^
        self.schannels = self.audata.channels
        self.srate = self.audata.samplerate
        self.dur = self.audata.duration
        self.nframes = self.srate * self.dur
        self.nsamples = self.nframes * self.schannels
        self.chunk = len(self.buf)/self.swidth
        self.binSize = self.srate/float(self.chunk)
        self.window = np.blackman(self.chunk)

    #########################
    # Utils & Stats
    #########################

    def loseLast(self):
        # Ignore last chunk since it has fewer bins
        self.allF0 = self.allF0[0:len(self.allF0)-2]
        self.allEnergy = self.allEnergy[0:len(self.allEnergy)-2]
        self.allCentroid = self.allCentroid[0:len(self.allCentroid)-2]
        self.allSkewness = self.allSkewness[0:len(self.allSkewness)-2]
        self.allKurtosis = self.allKurtosis[0:len(self.allKurtosis)-2]
        self.FFT = self.FFT[0:len(self.FFT)-2]

    # Convert fft bin index to its corresponding frequency
    def indexToFreq(self, index):
        return index*self.binSize
        
    def freqToIndex(self, freq):
        return freq/self.binSize
    
    def calculateF0(self, fft):
        freq = float('nan')
        f0Index = fft[1:].argmax()+1 # find maximum-energy bin
        
        # interpolate around max-energy freq unless f0 is the last bin :/
        if f0Index != len(fft)-1:
            y0, y1, y2 = np.log(fft[f0Index-1:f0Index+2:])
            x1 = (y2 - y0) * .5 / (2 * y1 - y2 - y0)
            freq = self.indexToFreq(f0Index + x1)
        else:
            freq = self.indexToFreq(f0Index)
            
        return freq
            
    def calculateSpectralCentroid(self, fft):
        centroidIndex = np.sum(np.arange(len(fft))*fft)/fft.sum()
        return self.indexToFreq(centroidIndex)
