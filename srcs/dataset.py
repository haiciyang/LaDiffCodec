import glob
import torch
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader


class EnCodec_data(Dataset):

	def __init__(self, path, task='train', seq_len_p_sec=5, sample_rate=16000, multi=False, n_spks=0): 

		# generate 5s fragments
		# self.path = '/home/v-haiciyang/data/haici/read_speech_16k/*'
		# self.path = os.environ.get("AMLT_DATA_DIR", ".") + 'read_speech_16k/*'
		
		# self.val_max = 46.58
		# self.train_max = 32.327

		self.path = path
		
		self.spks_l = glob.glob(self.path)
		if n_spks != 0:
			self.spks_l = self.spks_l[:n_spks]
		else:
			n_spks = len(self.spks_l)
		self.task = task
		self.seq_len_p_sec = seq_len_p_sec
		self.sample_rate = sample_rate
		self.multi = multi

		self.seg_std_max = []
		self.seg_std = []

		self.data_max =  152 #151.8879553636901

	def __len__(self):

		return len(self.spks_l)


	def __getitem__(self, idx):
			

		if self.multi:
			
			# idx = 0 # ***
			
			seq1, seg_id = self.get_seq(idx, None) # This seg_id is randomly picked
			idx2 = (idx + 1) % self.__len__()

			# idxr = 1 # *****
			seq2, _ = self.get_seq(idx2, seg_id) # Use the same seg_id

			return seq1 + seq2 #, np.vstack((seq1, seq2)), torch.tensor((idx, idx2))

		else:
			return self.get_seq(idx)					
		

	def get_seq(self, idx, seg_id=None):

		spk_folder = self.spks_l[idx]

		seg_l = glob.glob(spk_folder + '/*.pth') 
		len_seg = len(seg_l)
		train_num = len_seg - 2

		if self.task == 'train':
			if seg_id is None:
				seg_id = torch.randint(train_num, (1,)) # The only randomness
			else:
				seg_id = min(seg_id, train_num)
		elif self.task == 'valid':
			if seg_id is None:
				seg_id = -1
			else:
				seg_id = -2
		else:
			print('Task can only be train or valid.')
	
		seg = torch.load(seg_l[seg_id])

		if self.seq_len_p_sec < 5:

			seq_length = int(self.seq_len_p_sec * 16000)
			while 1:
				loc = torch.randint(len(seg)-seq_length, (1,))
			# loc = 0 # ****
				selected_seg = seg[loc: loc + seq_length]
				if not np.isclose(np.std(selected_seg), 0): # exclude empty sampels
					seg = selected_seg
					break
					# print('Contains empty segments')

		seg = seg / 32768
		
		
		# Normalize and add random gain
		# seg = seg / (np.std(seg) + 1e-8) 
		# seg = seg / self.data_max

		
		# seg /= np.max(seg) + 1e-8
		# gain = np.random.randint(-10, 7, (1,))
		# scale = np.power(10, gain/20)
		# seg *= scale

		return seg # , seg_id


	def compute_max(self):
		smax = 0
		for spk_folder in tqdm(self.spks_l):
			seg_l = glob.glob(spk_folder + '/*.pth') 
			for seg in seg_l:
				s = torch.load(seg)
				if not np.isclose(np.std(s), 0): # exclude empty sampels
					s = s / np.std(s)
				smax = max(smax, max(abs(s)))
		return smax
		
















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

		# return seg









