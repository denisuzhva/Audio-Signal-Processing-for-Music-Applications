import numpy as np
from scipy.signal import get_window
from scipy.fftpack import fft
import sys, os, math
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import utilFunctions as UF
import hprModel as HPR
import stft as STFT


filename = '260559__roganderrick__liquor-bottle-pour-01.wav'
#filename ='speech-female.wav'
fs, x = UF.wavread(filename)
window = 'blackman'
M = 4001
N = 4096
t = -100
minf0 = 50
maxf0 = 300
f0et = 5
minSineDur = 0.1
harmDevSlope = .001
nH = 100


Ns = 512
H = Ns // 4

w = get_window(window, M)
hfreq, hmag, hphase, xr = HPR.hprModelAnal(x, fs, w, N, H, t, minSineDur, nH, minf0, maxf0, f0et, harmDevSlope)
mXr, pXr = STFT.stftAnal(xr, w, N, H)
y, yh = HPR.hprModelSynth(hfreq, hmag, hphase, xr, Ns, H, fs)

outputFileSines = os.path.basename(filename)[:-4] + 'harmonic.wav'
outputFileResidual = os.path.basename(filename)[:-4] + 'stochastic.wav'
outputFile = os.path.basename(filename)[:-4] + 'reconstructed.wav'

UF.wavwrite(yh, fs, outputFileSines)
UF.wavwrite(xr, fs, outputFileResidual)
UF.wavwrite(y, fs, outputFile)
