from tqdm import tqdm
import reservoirpy as rpy
from reservoirpy.nodes import ESN
from reservoirpy.nodes import Reservoir, Ridge, Input
from reservoirpy.observables import nrmse, rsquare
from reservoirpy.hyper import research
import json
import numpy as np

from sklearn.metrics import accuracy_score, f1_score
from jiwer import wer

from helper_functions import *  # helper functions for RC models

### load data ###

from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import OneHotEncoder

from load_data import *

### One-hot Encoding ###

one_hot_p = OneHotEncoder(categories=[phoneme], sparse_output=False)
Y_p = [one_hot_p.fit_transform(np.array(y)) for y in Y_p]

### RC model Classes ###

def calculate_acc(goldstandards, predictions, decoder=one_hot_p, decoder_dict=phoneme):
    '''
    Calculate Accuracy, F1 score, WER, and
    Insertions, Deletions, Substitutions details
    '''
    FACC = []
    F1 = []
    WER = []
    IDS = []
    for y_t, y_p in zip(goldstandards, predictions):
        targets = np.vstack(decoder.inverse_transform(y_t)).flatten()
    
        top_1 = np.argmax(y_p, axis=1)
        top_1 = np.array([decoder_dict[t] for t in top_1])
    
        accuracy = accuracy_score(targets, top_1)
        f1 = f1_score(targets, top_1, average="macro")        
        predict_phs = flat_n_get_string(top_1); target_phs = flat_n_get_string(targets)
        werate = wer(target_phs, predict_phs)
        
        FACC.append(accuracy)
        F1.append(f1)
        WER.append(werate)

        # observe operations like insertion, deletion, and substitution
        editops = decode_levenshtein(predict_phs, target_phs)
        IDS.append(editops)

    return [FACC, F1, WER, IDS]

class single_esn():
    '''
    Create and train a single ESN (reservoir -> readout)
    '''
    
    def __init__(self, units=1000, leak_rate=0.2, spectral_radius=0.4,
                 inputs_scaling=0.8, connectivity=0.5, input_connectivity=0.8,
                 regularisation=1e-5, seed=1234):
        '''
        Initialise an ESN
        '''
        seed = seed
        reservoir = Reservoir(units, sr=spectral_radius,
                          lr=leak_rate, rc_connectivity=connectivity,
                          input_connectivity=input_connectivity, seed=seed)
    
        readout = Ridge(ridge=regularisation)

        self.model = ESN(reservoir=reservoir, readout=readout, workers=-1)

    def train(self, X_train, y_train):
        '''
        Train the ESN
        '''
        self.model = self.model.fit(X_train, y_train)

    def evaluation(self, X_test, y_test, decoder=one_hot_p, decoder_dict=phoneme):
        '''
        Test/run the ESN and evaluate using ACC
        Return ACC scores
        '''
        self.outputs = self.model.run(X_test)
        self.scores = calculate_acc(y_test, self.outputs, decoder=decoder, decoder_dict=decoder_dict)

        print("Average Frame accuracy:", f"{np.mean(self.scores[0]):.4f}", "±", f"{np.std(self.scores[0]):.5f}")
        print("Average F1 score: ", f"{np.mean(self.scores[1]):.4f}", "±", f"{np.std(self.scores[1]):.5f}")
        print("Average WER: ", f"{np.mean(self.scores[2]):.4f}", "±", f"{np.std(self.scores[2]):.5f}")

    def __del__(self):
        print(f"Single ESN Object {self.model} is being destroyed.")

class single_esn_v1(single_esn):
    '''
    Create a single ESN with dual input:
    [data, data >> reservoir] >> readout 
    '''
    def __init__(self, units=1000, leak_rate=0.2, spectral_radius=0.4,
                 inputs_scaling=0.4, connectivity=0.5, input_connectivity=0.8,
                 regularisation=1e-5, seed=1234):
        '''
        Initialise an ESN
        '''
        seed = seed
        
        data = Input()
        reservoir = Reservoir(units, sr=spectral_radius,
                          lr=leak_rate, rc_connectivity=connectivity,
                          input_connectivity=input_connectivity, seed=seed)
        readout = Ridge(ridge=regularisation)

        self.model = data >> reservoir >> readout & data >> readout


class hrc(single_esn):
    '''
    Create and train a hierarical RC (reservoir -> readout -> reservoir -> readout)
    The parameter *height* specifies the number of ESN layer you want to create (one layer = reservoir -> readout)
    '''
    def __init__(self, height, units=1000, leak_rate=0.2, spectral_radius=0.4,
                 inputs_scaling=0.4, connectivity=0.5, input_connectivity=0.8,
                 regularisation=1e-5, seed=2468):
        '''
        Initialise a hierarical RC
        '''
        reservoir = Reservoir(units, sr=spectral_radius,
                              lr=leak_rate, rc_connectivity=connectivity,
                              input_connectivity=input_connectivity, seed=seed)
    
        readout = Ridge(ridge=regularisation)
        self.model = reservoir >> readout
        #self.model = ESN(reservoir=reservoir, readout=readout, workers=-1)
        
        for i in range(height-1):
            reservoir = Reservoir(units, sr=spectral_radius,
                              lr=leak_rate, rc_connectivity=connectivity,
                              input_connectivity=input_connectivity, seed=seed)
    
            readout = Ridge(ridge=regularisation)
            self.model = self.model >> reservoir >> readout
            #self.model = self.model & ESN(reservoir=reservoir, readout=readout, workers=-1)

    def train(self, X_train, y_train):
        '''
        Train the ESN
        '''
        y1 = y2 = y_train
        self.model = self.model.fit(X_train, {self.model.node_names[1]:y1, self.model.node_names[3]:y2})
