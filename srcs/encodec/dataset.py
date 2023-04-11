import os
import glob
import torch
import random
import numpy as np
from scipy import signal
from scipy.io import wavfile
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader


class EnCodec_data(Dataset):

	def __init__(self, path, task='train', seq_len_p_sec=5, sample_rate=16000, multi=False): 

		# generate 5s fragments
		# self.path = '/home/v-haiciyang/data/haici/read_speech_16k/*'
		# self.path = os.environ.get("AMLT_DATA_DIR", ".") + 'read_speech_16k/*'
		self.path = path
		self.spks_l = glob.glob(self.path)
		self.task = task
		self.seq_len_p_sec = seq_len_p_sec
		self.sample_rate = sample_rate
		self.multi = multi
		self.max_seg = 0

	def __len__(self):

		return len(self.spks_l)


	def __getitem__(self, idx):

		if self.multi:

			seq1 = self.get_seq(idx)
			if self.task =='train':
				idxr = torch.randint(self.__len__(), (1,))
			else:
				idxr = (idx + 20) % self.__len__()

			seqr = self.get_seq(idxr)

			output = seq1 + seqr

		else:

			output = self.get_seq(idx)
		
		if self.sample_rate != 16000:
			output = output.squeeze().data.numpy() # (80000)
			output = signal.resample(output, self.sample_rate*self.seq_len_p_sec)
			output = torch.tensor(output).unsqueeze(0)
		
		return output


	def get_seq(self, idx):

		length = len(self.spks_l)
		spk_folder = self.spks_l[idx]

		seg_l = glob.glob(spk_folder + '/*.pth') # Leave the rest for eval and test

		len_seg = len(seg_l)
		valid_num = len_seg//10 + 1
		train_num = len_seg - valid_num

		
		if self.task == 'train':
			seg_id = torch.randint(train_num, (1,))
		elif self.task == 'valid':
			seg_id = torch.randint(valid_num, (1,))
			seg_id = - (seg_id + 1) # -1 or -2
		elif self.task == 'eval':
			seg_id = -2
		else:
			print('Task can only be train or valid.')

		seg = torch.load(seg_l[seg_id])

		# Normalize and add random gain
		seg = seg / (np.std(seg) + 1e-20)
		# self.max_seg = max(max(seg), self.max_seg)
		# seg /= np.max(seg)
		
		gain = np.random.randint(-10, 7, (1,))
		scale = np.power(10, gain/20)
		seg *= scale


		return seg
















			# if self.task == 'valid':
			# 	sample_id = torch.randint(2, (1,)) # 0 or 1
			# 	sample_id = - (sample_id + 1) # -1 or -2
			# else:
			# 	sample_id = torch.randint(max(1, len(sample_l)-2), (1,))
			
			# try:
			# 	wav_fname = sample_l[sample_id]
			# 	samplerate, sample = wavfile.read(wav_fname)
			# except Exception as e:
			# 	idx = torch.randint(self.__len__(), (1,))
			# 	continue
			# else:
			# 	pass
			# finally:
			# 	pass
			

			# seg_id = torch.randint(30 - self.seq_len_p_sec, (1,)) # Each sample contains 30s speech
			# seg = sample[seg_id * self.sample_rate :(seg_id + self.seq_len_p_sec) * self.sample_rate]

			# if len(seg) == self.sample_rate * self.seq_len_p_sec:
			# 	break

		return seg









