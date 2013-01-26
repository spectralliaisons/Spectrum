from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

cc = lambda arg: colorConverter.to_rgba(arg, alpha=0.6)

import Spectrum
sound = Spectrum.Analyze('blub.wav')
fft = sound.FFT
ln = len(fft)

xs = np.arange(1025)
verts = []
ccs = []
zs = np.arange(ln)
maxe = np.max(fft[0])

print "MAX E %d" % maxe

for chunk in fft:
    #ys[0], ys[-1] = 0, 0 # smooth at ends
    verts.append(zip(zs, chunk))
    #ccs.append(cc('g'))

    if np.max(chunk) > maxe:
        maxe = np.max(chunk)

poly = PolyCollection(verts)
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=zs, zdir='y')

ax.set_xlabel('Frequency')
ax.set_xlim3d(0, 600)
ax.set_ylabel('Energy')
ax.set_ylim3d(0, maxe)
ax.set_zlabel('Time')
ax.set_zlim3d(0, ln)

plt.show()