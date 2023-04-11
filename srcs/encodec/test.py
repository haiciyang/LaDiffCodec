import glob

path = '/home/v-haiciyang/data/haici/read_speech_16k/*'

files = glob.glob(path)

il = []

for f in files:
	t = glob.glob(f+'/*')
	il.append(len(t))
	if len(t) < 1:
		print(t)
print(max(il), min(il))


