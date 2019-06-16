from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy as np
from scipy.stats import energy_distance
import sys

(rate1,sig1) = wav.read(sys.argv[1])
mfcc_feat1 = mfcc(sig1,rate1)
fbank_feat1 = logfbank(sig1,rate1)

(rate2,sig2) = wav.read(sys.argv[2])
mfcc_feat2 = mfcc(sig2,rate2)
fbank_feat2 = logfbank(sig2,rate2)

print('mfcc dist:', np.sqrt(np.sum((mfcc_feat1[:,None] - mfcc_feat2) ** 2)))
print('fbank dist:', energy_distance(fbank_feat1.flatten(), fbank_feat2.flatten()))
