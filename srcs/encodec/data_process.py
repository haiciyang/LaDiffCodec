import os
import glob
import torch
from tqdm import tqdm
from scipy.io import wavfile

path = '/home/v-haiciyang/data/haici/read_speech_16k/*'
seq_l = 16000*5


for spk_path in tqdm(glob.glob(path)):

	spk_id = spk_path.split('/')[-1]

	save_spk_path = '/home/v-haiciyang/data/haici/dns_pth/' + spk_id
	if not os.path.isdir(save_spk_path):
		os.makedirs(save_spk_path)

	for sample_path in glob.glob(spk_path + '/*'):
		
		sample_id = sample_path.split('_')[-1][:-4]

		try:
			samplerate, sample = wavfile.read(sample_path)

			for idx, seq_loc in enumerate(range(0, len(sample)-seq_l, seq_l)):

				torch.save(sample[seq_loc : seq_loc+seq_l], f'/home/v-haiciyang/data/haici/dns_pth/{spk_id}/{sample_id}_{idx}.pth')
		except Exception as e:
			continue









