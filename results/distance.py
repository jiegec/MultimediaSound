from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy as np
from scipy.stats import energy_distance
import sys

(rate1,sig1) = wav.read(sys.argv[1])
A = mfcc(sig1,rate1)
fbank_feat1 = logfbank(sig1,rate1)

(rate2,sig2) = wav.read(sys.argv[2])
B = mfcc(sig2,rate2)
fbank_feat2 = logfbank(sig2,rate2)

if A.shape[0] < B.shape[0]:
    A = np.vstack((A, np.zeros((B.shape[0] - A.shape[0], 13))))
elif A.shape[0] > B.shape[0]:
    B = np.vstack((B, np.zeros((A.shape[0] - B.shape[0], 13)))) 

print('mfcc dist:', np.sqrt(np.sum((A - B) ** 2)))
print('fbank dist:', energy_distance(fbank_feat1.flatten(), fbank_feat2.flatten()))
