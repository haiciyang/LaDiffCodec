import os
import glob
import torch
import random
import librosa
import numpy as np
import scipy.io.wavfile as wavfile
# from config import ex
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader


class Dataset_Libri(Dataset):
    
    # @ex.capture
    def __init__(self, task = 'train', seq_len_p_sec=5, data_folder_path='/data/hy17/librispeech/librispeech'): # 28539
        
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
        
        if self.task == 'train':
            path = data_folder_path + '/train-clean-100/*/*/*.wav'
        elif self.task == 'valid' or self.task == 'eval':
            path = data_folder_path + '/dev-clean/*/*/*.wav'
        
        self.files = glob.glob(path)[:10000]

        self.signal_max = []
        self.length_sec = []
        

        print('Using original data')
        
    def __len__(self):
        
        return len(self.files)
    
    def normalize_data(self, x):

        # x = x / 32768
        x = x / max(abs(x)+1e-20)
        return x

    def __getitem__(self, idx):
        
        eps = 1e-10
    
        file_path = self.files[idx] # /media/sdb1/Data/librispeech/train-clean-100/103/1240/103-1240-0000.wav
        # print(file_path)
        sample_name = file_path.split('/')[-1][:-4] # 103-1240-0000
        
        # -- Load data -- 
        # in_data, sr = librosa.load(file_path, sr=None)
        sr, in_data  = wavfile.read(file_path)
        # in_data = in_data / 32768

        in_data = self.normalize_data(in_data)

 
        seq_length = int(self.seq_len_p_sec * 16000)
        
        if self.task == 'eval':
            loc = 0
            seg = in_data[loc: loc + seq_length]
        else:
            while len(in_data) < seq_length or np.isclose(np.std(in_data), 0):
                idx = (idx + 1) % self.__len__()
                file_path = self.files[idx]
                sr, in_data  = wavfile.read(file_path)
                in_data = self.normalize_data(in_data)

            while 1:
                if len(in_data) == seq_length:
                    loc = 0
                else:
                    loc = torch.randint(len(in_data)-seq_length, (1,))
                selected_seg = in_data[loc: loc + seq_length]
                if not np.isclose(np.std(selected_seg), 0): # exclude empty sampels
                    seg = selected_seg
                    break
                    # print('Contains empty segments')

        
        return seg
        
        
        
        
        
         
        