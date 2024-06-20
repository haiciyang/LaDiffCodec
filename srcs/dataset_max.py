import os
import glob
import torch
import random
import librosa
import numpy as np
import torchaudio
import torchaudio.transforms as T
import scipy.io.wavfile as wavfile
from torch.utils.data import Dataset, DataLoader

def normalize_data(x):
        # x = x / 32768
        x = x / max(abs(x)+1e-20) # the original normalization of ladiff
        return x

def standardize_data(x):
        x = x / (np.std(x) + 1e-10)
        return x

def is_clipping(x, clipping_threshold = 0.99):
    return max(abs(x)) > clipping_threshold

def db_gain_in_amp(db):
    return 10 ** (db / 20)
    
def standardize_random_gain(x, db_gain_l=-10, db_gain_h=6):
        x = standardize_data(x)
        return x
        # amp_gain = db_gain_in_amp(random.randint(db_gain_l, db_gain_h))
        # return x * amp_gain

class Dataset_Max(Dataset):
    
    # @ex.capture
    def __init__(self, task = 'train', seq_len_p_sec=5, folder_list=[], model_sampling_rate=16000, data_proc='norm' ): #, data_folder_path='/data/hy17/librispeech/librispeech'): # 28539
        
        '''
        # training - 28539
        # testing - 2703
        # One "chunk" corresponds to 2400 samples, 15 frames
        # qtz: 
            -1: return all unquantized features
             0: return unquantized ceps features and quantized pitches
             1: return all quantized features     
        '''
        
        self.task = task
        self.seq_len_p_sec = seq_len_p_sec
        self.model_sampling_rate = model_sampling_rate
        self.data_proc = data_proc
        
        self.files = []
        for path in folder_list:
            self.files += glob.glob(os.path.join(path, '**/*.wav'), recursive=True)

        self.signal_max = []
        self.length_sec = []
        
        print('Using original data')
        
    def __len__(self):
        
        return len(self.files)

    def __getitem__(self, idx):
    
        file_path = self.files[idx] # /media/sdb1/Data/librispeech/train-clean-100/103/1240/103-1240-0000.wav
        sample_name = file_path.split('/')[-1][:-4] # 103-1240-0000
        seq_length = int(self.seq_len_p_sec * self.model_sampling_rate)
    
        # -- Load data -- 
        # in_data, sr = torchaudio.load(file_path) # in_data [1, L]
        sr, in_data  = wavfile.read(file_path)
        # sr, in_data  = wavfile.read('/data/hy17/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav')
        # print(sr, in_data.shape)  
        # print(max(abs(in_data)))      
        # sr, in_data  = wavfile.read('/data/hy17/quesst14Database/Audio/quesst14_12480.wav')
        # print(sr, in_data.shape)
        # print(max(abs(in_data)))
        # sr, in_data  = wavfile.read('/data/hy17/voxceleb/dev_wav/id10734/dFyHCZVZNbI/00001.wav')
        # print(sr, in_data.shape)
        # print(max(abs(in_data)))
        # fake()
        
        # in_data = in_data[0]

        # while len(in_data) < seq_length or torch.isclose(torch.std(in_data), torch.tensor(0.0)) or is_clipping(in_data):
        while len(in_data) < seq_length or np.isclose(np.std(in_data), 0) or is_clipping(in_data, clipping_threshold=0.99 * 32768):
            idx = (idx + 1) % self.__len__()
            file_path = self.files[idx]
            sr, in_data  = wavfile.read(file_path)
            
        # if sr != self.model_sampling_rate:
        #     # print(file_path)
        #     in_data = torchaudio.functional.resample(in_data, sr, self.model_sampling_rate)
        
        if self.data_proc == 'norm':
            in_data = normalize_data(in_data)
        elif self.data_proc == 'standardize':
            in_data = standardize_random_gain(in_data)
        else:
            raise ValueError('Invalid data process')
        
        if self.task == 'eval':
            loc = 0
            seg = in_data[loc: loc + seq_length]
        else:
            while 1:
                if len(in_data) == seq_length:
                    loc = 0
                else:
                    loc = torch.randint(len(in_data)-seq_length, (1,))
                selected_seg = in_data[loc: loc + seq_length]
                # if not torch.isclose(torch.std(selected_seg), torch.tensor(0.0)): # exclude empty sampels
                if not np.isclose(np.std(selected_seg), 0): # exclude empty sampels
                    seg = selected_seg
                    break
        return seg
        
        
        
        
        
         
        