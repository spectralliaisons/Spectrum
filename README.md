Spectrum
Wes Jackson Dec 2012-
========

Spectral Analysis in Python
-> Analyzes spectrum across entire sound file

* supports many file types; all converted into mono wav format since analysis can vary depending on encoding
* 'What is the sound's characteristic frequency?'
		-> calculates 'weighted spectral centroid': weighted by total energy of sound across the duration of the entire sound 
* graphs spectrograms across time by energy and spectral change (std of a frequency bin as its energy changes over the duration of the sound)