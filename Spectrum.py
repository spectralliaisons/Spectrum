# The MIT License (MIT)

# Copyright (c) 2013 Wesley Jackson

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

############################

 Wes Jackson 
 feloniousriot@gmail.com
 Dec 2012


 
 Spectral analysis for many types of audio file
 All input converted to mono wav, as analysis may differ depending on file format

 USAGE: 
 import Spectrum as s
 a = s.Analyze('sound/kombucut.wav', maxFreq=8)
 a.plot()

 TODO:
 [] Credit sources where I found useful code
 [] Sounds are converted to mono wav but sometimes the frequencies determined are off by a factor of 0.5X. Why is this?  

 Requirements: pydub

############################

import math
import wave
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
import matplotlib.axis as Axis
import warnings
import struct
from pydub import AudioSegment as AS # for converting files to standard format

class Analyze:
    def __init__(self, 
        fileName,
        maxFreq=12., #kHz
        windowSize=2048, 
        zeroPad=False, 
        window='hanning'):
        
        self.window = window
        self.doZeroPad = zeroPad
        self.fMax = maxFreq
        self.windowSize = windowSize

        # convert to mono wav
        self.fileName = self.exportMonoWav(fileName)

        print self.fileName

        # fft
        self.analyze()  

    #########################   
    # Spectral Analysis
    #########################
    
    def analyze(self):

        # clear & configure based on file format
        self.clear()
        self.configure()
        
        # grab some audio frames
        self.updateBuf()

        while len(self.buf) == self.windowSize*self.swidth:
            self.fftBuf()
            self.updateBuf()

        # combine analyses across frames
        self.generalAnalysis()

    def updateBuf(self):
        self.buf = self.audata.readframes(self.windowSize)

    def clear(self):
        self.allF0 = []
        self.allEnergy = []
        self.allCentroid = []
        self.allSkewness = []
        self.allKurtosis = []
        self.FFT = []
        self.freqRange = []

    def configure(self):
        self.audata = wave.open(self.fileName, 'rb') # audioread.audio_open(self.fileName)

        self.swidth = self.audata.getsampwidth() # 2 # 16-bit ^^ CHECK THIS IT WAS 2
        self.schannels = self.audata.getnchannels() # self.audata.channels
        self.srate = self.audata.getframerate() # self.audata.samplerate

        # for zero-padding, take fft size next power of 2 above window size
        self.paddedFFTSize = int(2**(1+np.ceil(np.log2(2*self.windowSize))))

        print "Width, Channels, Rate, windowSize %d, %d, %d, %d" % (self.swidth, self.schannels, self.srate, self.windowSize)
        #print "windowSize, paddedFFTSize: %d, %d" % (self.windowSize, self.paddedFFTSize)
    
    #####################################################  
    # Run FFT for a chunk of audio and compile analysis
    #####################################################

    def fftBuf(self):

        ### 1. DECODE SAMPLES BUFFER

        fmt = "%dh" % int(len(self.buf)/self.swidth)

        #print "fmt size %d, %d" % (struct.calcsize(fmt), len(self.buf))

        indata = np.array(struct.unpack(fmt, self.buf))

        ### 2. WINDOW

        indata = self.applyWindow(indata, self.window)

        # width in kHz of e/a frequency bin
        self.binSize = (self.srate/(len(indata)/float(self.swidth)))/1000.

        ### 3. FFT
        if self.doZeroPad:
            realFFT = abs(np.fft.rfft(indata, n=self.paddedFFTSize))**2.
        else:
            realFFT = abs(np.fft.rfft(indata))**2.

        ### 4. Filter

        realFFT = realFFT[0:self.freqToIndex(self.fMax)]
            
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

        #print "Done!"
        #print "Unweighted, weighted mean:"
        print ">> F0: %f, %f" % (self.MeanF0, self.WeightedF0)
        #print ">> Energy: %f" % self.MeanEnergy
        print ">> Spectral Centroid: %f, %f" % (self.MeanSpectralCentroid, self.WeightedSpectralCentroid)
        #print ">> Skewness: %d, %d" % (self.MeanSkewness, self.WeightedSkewness)
        #print ">> Kurtosis: %d, %d" % (self.MeanKurtosis, self.WeightedKurtosis)

    #########################
    # Utils & Stats
    #########################

    def getFileType(self, str):
        return str[str.index('.')+1:]

    def getFileName(self, str):
        return str[0:str.index('.')]

    # Ensure standard format 
    def exportMonoWav(self, fileName):
        ext = self.getFileType(fileName)
        if ext == 'wav':
            pre = AS.from_wav(fileName)
        elif ext == 'mp3':
            pre = AS.from_mp3(fileName)
        elif ext == 'ogg':
            pre = AS.from_ogg(fileName)
        elif ext == 'flv':
            pre = AS.from_flv(fileName)
        else:
            pre = AS.from_file(fileName)

        # set mono & 
        pre = pre.set_channels(1)
        #pre = pre.set_frame_rate(22050)

        fout = self.getFileName(fileName) + '_AS_MONO_WAV_44100.wav'
        pre.export(fout, format='wav')

        return fout

    def applyWindow(self, samples, window='hanning'):

        if window == 'bartlett':
            return samples*np.bartlett(len(samples))
        elif window == 'blackman':
            return samples*np.hanning(len(samples))
        elif window == 'hamming':
            return samples*np.hamming(len(samples))
        elif window == 'kaiser':
            return samples*np.kaiser(len(samples))
        else:
            return samples*np.hanning(len(samples))

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
        return index*float(self.binSize)
        
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
        centroidIndex = np.sum((1+np.arange(len(fft)))*fft)/float(fft.sum()) # +1 so index 0 isn't 0
        return self.indexToFreq(centroidIndex-1)

    def dB(self, a, b):
        return 10. * np.log10(a/b)

    # 1.567 -> 1.6
    def round(n):
        return round(n*10)/10.

    #########################
    # Visualization
    #########################

    ### Overlay audio frame spectrograms:
    ### 1. Linear freq by energy
    ### 2. Log freq by dB
    ### 3. Log freq by amount of change between audio frames
    def plot(self, xMin='NaN', xMax='NaN', dBMin=-90):

        plt.figure()
        plt.suptitle(self.fileName + ': F0: ' + str(int(self.WeightedF0)) + ', Centroid: ' + str(int(self.WeightedSpectralCentroid)))
        
        #########################
        # 1. Linear energy scale
        #########################

        linPlot = plt.subplot(311)

        if xMin == 'NaN':
            xMin = self.WeightedF0 - 0.05

        if xMax == 'NaN':
            xMax = self.fMax

        # x-Axis as frequency
        fs = []
        for f in range(int(self.freqToIndex(self.fMax))):
            fs.append(self.indexToFreq(f))

        # y-Axis as normalized energy
        for fft in self.FFT:
            ys = fft/np.amax(self.FFT)

            plt.plot(fs, ys, linewidth=2, color='black')
            plt.fill_between(fs, ys, facecolor='green', alpha=0.5)

        # plot centroid & fundamental freq
        f0 = plt.plot([self.WeightedF0], [self.MeanEnergy], 'b^')
        cent = plt.plot([self.WeightedSpectralCentroid], [self.MeanEnergy], 'ro')
        plt.setp(f0, 'markersize', 12.0, 'markeredgewidth', 2.0)
        plt.setp(cent, 'markersize', 12.0, 'markeredgewidth', 2.0)
        plt.title('All Audio Frames: Linear')
        #plt.text(0, 1, 'F0: ' + str(int(self.WeightedF0)) + ' Centroid: ' + str(int(self.WeightedSpectralCentroid)))
        plt.grid(True)

        #plt.xlabel('Frequency')
        plt.ylabel('Energy')
        plt.axis([xMin, xMax, 0, 1])
        linPlot.xaxis.set_major_formatter(FormatStrFormatter('%.01f'))
        linPlot.xaxis.set_minor_formatter(FormatStrFormatter('%.01f'))

        #########################
        # 2. dB energy scale
        #########################

        dBPlot = plt.subplot(312)

        # x-Axis as frequency
        fs = []
        for f in range(int(self.freqToIndex(self.fMax))):
            fs.append(self.indexToFreq(f))

        #mdB = self.dB(self.MeanEnergy, np.amax(self.FFT))
        alldBs = []
        # y-Axis as normalized energy
        for fft in self.FFT:
            #ys = fft/np.amax(self.FFT)
            dBs = []
            for i in fft:
                dB = max(dBMin, self.dB(i, np.amax(self.FFT)))
                dBs.append(dB)
                alldBs.append(dBs)

            #dBPlot.plot(fs, dBs, linewidth=2, color='black')
            plt.semilogx(fs, dBs, linewidth=2, color='black')
            plt.fill_between(fs, dBs, dBMin, facecolor='green', alpha=0.3)

        mindB = np.amin(alldBs)
        mdB = np.mean(alldBs)

        # plot centroid & fundamental freq
        f0 = dBPlot.plot([self.WeightedF0], [mdB], 'b^')
        cent = dBPlot.plot([self.WeightedSpectralCentroid], [mdB], 'ro')
        plt.setp(f0, 'markersize', 12.0, 'markeredgewidth', 2.0)
        plt.setp(cent, 'markersize', 12.0, 'markeredgewidth', 2.0)

        plt.title('All Audio Frames: dB')
        plt.grid(True)

        #plt.xlabel('Frequency')
        plt.ylabel('dB')
        plt.axis([xMin, xMax, mindB, 0])
        #plt.xscale('log')
        dBPlot.xaxis.set_major_formatter(FormatStrFormatter('%.01f'))
        dBPlot.xaxis.set_minor_formatter(FormatStrFormatter('%.01f'))
        #########################
        # 3. Spectral change as stdev of dB values for a freq bin across audio frames
        # Use dB since much more energy at low freqs means higher stdev
        #########################

        devPlot = plt.subplot(313)

        # e/a freq bin as array of energy in e/a frame
        numBins = len(self.FFT[0])
        numFrames = len(self.FFT)
        allBins = np.arange(numBins*numFrames).reshape(numBins, numFrames)
        binDev = np.arange(numBins)

        for bin in range(numBins):
            for frame in range(numFrames):
                allBins[bin][frame] = self.dB(self.FFT[frame][bin], np.amax(self.FFT))

            binDev[bin] = np.std(allBins[bin])

        #normalize
        #binDev = binDev/float(np.amax(binDev))

        plt.semilogx(fs, binDev, linewidth=2, color='black')
        plt.fill_between(fs, binDev, facecolor='red', alpha=0.5)

        # plot centroid & fundamental freq
        f0 = plt.plot([self.WeightedF0], [min(binDev)+5], 'b^')
        cent = plt.plot([self.WeightedSpectralCentroid], [min(binDev)+5], 'ro')
        plt.setp(f0, 'markersize', 12.0, 'markeredgewidth', 2.0)
        plt.setp(cent, 'markersize', 12.0, 'markeredgewidth', 2.0)

        plt.title('Spectral Deviation Across Audio Frames')
        plt.grid(True)

        plt.xlabel('Frequency (kHz)')
        plt.ylabel('STD (dB)')
        plt.axis([xMin, xMax, min(binDev), max(binDev)])
        devPlot.xaxis.set_major_formatter(FormatStrFormatter('%.01f'))
        devPlot.xaxis.set_minor_formatter(FormatStrFormatter('%.01f'))

        plt.show()