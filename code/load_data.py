import os
import glob
import math
import numpy as np
import pandas as pd
import textgrid
import librosa as lbr
from tqdm import tqdm

win_length = 512
n_fft = 1024      
hop_length = 512  
fmin = 200
fmax = 8000
lifter = 40
n_mfcc = 13

def downsized_phones(old_phone):
    if old_phone == "sp" or old_phone == "sil" or old_phone == "":
        return "__SIL__"
    elif old_phone == "spn":
        return "__UNK__"
    else:
        return ''.join(ph for ph in old_phone if not ph.isdigit())

def rescaling_data(X):
    flatten_data = [item for sent in X for item in sent]
    mean = np.mean(flatten_data)
    std = np.std(flatten_data)

    rescaled_data = []
    for sent in tqdm(X):
        flat = sent.flatten()
        norm = (flat-flat.min())/(flat.max() - flat.min())
        restore = norm.reshape(sent.shape)
        rescaled_data.append(restore)

    return rescaled_data

def load_processed_data(directory="./librespeech_360/train-clean-360", max_songs=600, sampling_rate=44100):
    audios = sorted(glob.glob(directory + "/**/*.flac", recursive=True))
    annotations = sorted(glob.glob(directory + "/**/*.TextGrid", recursive=True))

    X = []
    Y_p = []
    vocab = set()
    phoneme = set()

    max_songs = min(len(audios), max_songs)
    
    # STEP 1. load and process audio data and annotations and seperate by frames
    for audio, annotation, _ in tqdm(zip(audios, annotations, range(max_songs)), total=max_songs):
        tg = textgrid.TextGrid.fromFile(annotation)

        try: 
            # process audio (resampling, get MFCCs)
            wav, rate = lbr.load(audio, sr=sampling_rate)
            x = lbr.feature.mfcc(y=wav, sr=rate,
                                  win_length=win_length, hop_length=hop_length,
                                  n_fft=n_fft, fmin=fmin, fmax=fmax, lifter=lifter,
                                  n_mfcc=n_mfcc)
            delta = lbr.feature.delta(x, mode="wrap")
            delta2 = lbr.feature.delta(x, order=2, mode="wrap")
    
            X.append(np.vstack([x, delta, delta2]).T)
    
            yp = [["__SIL__"]] * x.shape[1]
    
            # process phonemes (71 phones -> 39+2 phones)
            for i in range(len(tg[0])):
                start = max(0, round(tg[0][i].minTime * rate / hop_length))
                end = min(x.shape[1], round(tg[0][i].maxTime * rate / hop_length))
                new_phone = downsized_phones(tg[0][i].mark)
                yp[start:end] = [[new_phone]] * (end - start)
                phoneme.add(new_phone)
                
            Y_p.append(yp)
        except:
            print(f"Error when processing file {audio}")
    
    # STEP 2. normalise audio data
    X_norm = rescaling_data(X)

    # STEP 3. get phone label dict
    #label_to_ix = {ph:i for i, ph in enumerate(phoneme)}

    return X_norm, Y_p, list(phoneme)

# 1. load and preprocess audio and label data
X, Y_p, phoneme = load_processed_data(max_songs=600)
